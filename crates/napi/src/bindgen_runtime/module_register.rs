use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::ptr;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::thread::ThreadId;

use once_cell::sync::Lazy;

use crate::{
  check_status, check_status_or_throw, sys, Env, JsError, JsFunction, Property, Result, Value,
  ValueType,
};

pub type ExportRegisterCallback = unsafe fn(sys::napi_env) -> Result<sys::napi_value>;
pub type ModuleExportsCallback =
  unsafe fn(env: sys::napi_env, exports: sys::napi_value) -> Result<()>;

struct PersistedPerInstanceVec<T> {
  inner: AtomicPtr<T>,
  length: AtomicUsize,
}

impl<T> Default for PersistedPerInstanceVec<T> {
  fn default() -> Self {
    let mut vec: Vec<T> = Vec::with_capacity(1);
    let ret = Self {
      inner: AtomicPtr::new(vec.as_mut_ptr()),
      length: AtomicUsize::new(0),
    };
    std::mem::forget(vec);
    ret
  }
}

impl<T> PersistedPerInstanceVec<T> {
  #[allow(clippy::mut_from_ref)]
  fn borrow_mut<F>(&self, f: F)
  where
    F: FnOnce(&mut [T]),
  {
    let length = self.length.load(Ordering::Relaxed);
    if length == 0 {
      f(&mut []);
    } else {
      let inner = self.inner.load(Ordering::Relaxed);
      let mut temp = unsafe { Vec::from_raw_parts(inner, length, length) };
      f(temp.as_mut_slice());
      // Inner Vec has been reallocated, so we need to update the pointer
      if temp.as_mut_ptr() != inner {
        self.inner.store(temp.as_mut_ptr(), Ordering::Relaxed);
      }
      self.length.store(temp.len(), Ordering::Relaxed);
      std::mem::forget(temp);
    }
  }

  fn push(&self, item: T) {
    let length = self.length.load(Ordering::Relaxed);
    let inner = self.inner.load(Ordering::Relaxed);
    let mut temp = unsafe { Vec::from_raw_parts(inner, length, length) };
    temp.push(item);
    // Inner Vec has been reallocated, so we need to update the pointer
    if temp.as_mut_ptr() != inner {
      self.inner.store(temp.as_mut_ptr(), Ordering::Relaxed);
    }
    std::mem::forget(temp);

    self.length.fetch_add(1, Ordering::Relaxed);
  }
}

unsafe impl<T: Send> Send for PersistedPerInstanceVec<T> {}
unsafe impl<T: Sync> Sync for PersistedPerInstanceVec<T> {}

pub(crate) struct PersistedPerInstanceHashMap<K, V>(*mut HashMap<K, V>);

impl<K, V> PersistedPerInstanceHashMap<K, V> {
  pub(crate) fn from_hashmap(hashmap: HashMap<K, V>) -> Self {
    Self(Box::into_raw(Box::new(hashmap)))
  }

  #[allow(clippy::mut_from_ref)]
  pub(crate) fn borrow_mut<F, R>(&self, f: F) -> R
  where
    F: FnOnce(&mut HashMap<K, V>) -> R,
  {
    f(unsafe { Box::leak(Box::from_raw(self.0)) })
  }
}

impl<K, V> Default for PersistedPerInstanceHashMap<K, V> {
  fn default() -> Self {
    let map = Default::default();
    Self(Box::into_raw(Box::new(map)))
  }
}

type ModuleRegisterCallback =
  PersistedPerInstanceVec<(Option<&'static str>, (&'static str, ExportRegisterCallback))>;

type ModuleClassProperty = PersistedPerInstanceHashMap<
  &'static str,
  HashMap<Option<&'static str>, (&'static str, Vec<Property>)>,
>;

unsafe impl<K, V> Send for PersistedPerInstanceHashMap<K, V> {}
unsafe impl<K, V> Sync for PersistedPerInstanceHashMap<K, V> {}

type FnRegisterMap =
  PersistedPerInstanceHashMap<ExportRegisterCallback, (sys::napi_callback, &'static str)>;
type RegisteredClassesMap = PersistedPerInstanceHashMap<ThreadId, RegisteredClasses>;

static MODULE_REGISTER_CALLBACK: Lazy<ModuleRegisterCallback> = Lazy::new(Default::default);
static MODULE_CLASS_PROPERTIES: Lazy<ModuleClassProperty> = Lazy::new(Default::default);
static IS_FIRST_MODULE: AtomicBool = AtomicBool::new(true);
static FIRST_MODULE_REGISTERED: AtomicBool = AtomicBool::new(false);
static REGISTERED_CLASSES: Lazy<RegisteredClassesMap> = Lazy::new(Default::default);
static FN_REGISTER_MAP: Lazy<FnRegisterMap> = Lazy::new(Default::default);
// Per-environment CustomGC registry.
//
// A `Buffer`/`TypedArray` received from JS carries a `napi_reference` bound to
// the V8 isolate of the environment it was created in. That reference can only
// be released on *that* environment's thread — V8 global handles are
// isolate-bound, and touching one from another isolate's thread is a fatal error
// (`Check failed: node->IsInUse()` in `GlobalHandles::MakeWeak`).
//
// When such a value is dropped on a non-owning thread (e.g. moved into an
// `async fn` future on a tokio worker, or sent to another thread), its unref is
// routed back to the owning environment's thread through that env's CustomGC
// `ThreadsafeFunction`. Each environment (the main thread and every
// `worker_thread`) registers its OWN tsfn here, keyed by its `napi_env` pointer.
//
// (Previously a single process-global tsfn was overwritten by every environment
// that registered, so an off-thread unref of a worker's Buffer was routed to
// whichever environment registered last — the wrong, or a torn-down, isolate.)
#[cfg(all(feature = "napi4", not(target_arch = "wasm32")))]
pub(crate) struct CustomGcTsfn(pub(crate) sys::napi_threadsafe_function);
// SAFETY: a `napi_threadsafe_function` is explicitly designed to be invoked from
// any thread (`napi_call_threadsafe_function`).
#[cfg(all(feature = "napi4", not(target_arch = "wasm32")))]
unsafe impl Send for CustomGcTsfn {}
#[cfg(all(feature = "napi4", not(target_arch = "wasm32")))]
unsafe impl Sync for CustomGcTsfn {}

#[cfg(all(feature = "napi4", not(target_arch = "wasm32")))]
pub(crate) static CUSTOM_GC_REGISTRY: Lazy<std::sync::RwLock<HashMap<usize, CustomGcTsfn>>> =
  Lazy::new(Default::default);

#[cfg(all(feature = "napi4", not(target_arch = "wasm32")))]
thread_local! {
  // The `napi_env` of the environment this thread belongs to, set when the
  // module registers on this thread. Null on non-env threads (tokio workers,
  // user-spawned threads). Used to decide whether a Buffer/TypedArray reference
  // can be released directly (we are on its owning env thread) or must be routed
  // through that env's CustomGC tsfn.
  pub(crate) static CURRENT_ENV: std::cell::Cell<sys::napi_env> =
    std::cell::Cell::new(ptr::null_mut());
}

/// Release a `napi_reference` held by a dropping `Buffer`/`TypedArray`.
///
/// `env` is the environment the reference was created in. If we are on that
/// environment's own thread the reference is released directly; otherwise the
/// release is routed to the owning env's thread through its CustomGC tsfn so the
/// release runs on the correct isolate.
#[cfg(all(feature = "napi4", not(target_arch = "wasm32")))]
pub(crate) fn route_custom_gc_unref(ref_: sys::napi_ref, env: sys::napi_env) {
  if ref_.is_null() || env.is_null() {
    return;
  }
  // On the owning environment's own thread: release directly. Best-effort —
  // a dropping value must never throw, and these can fail during env teardown.
  if CURRENT_ENV.with(|cell| cell.get()) == env {
    let mut ref_count = 0;
    let status = unsafe { sys::napi_reference_unref(env, ref_, &mut ref_count) };
    debug_assert!(
      status != sys::Status::napi_ok || ref_count == 0,
      "Buffer reference count in drop is not zero"
    );
    unsafe { sys::napi_delete_reference(env, ref_) };
    return;
  }
  // Off the owning thread: route the release to the owning env's CustomGC tsfn so
  // it runs on the correct isolate. Holding the read lock across the call keeps
  // the tsfn alive against concurrent env teardown — `custom_gc_finalize` takes
  // the write lock (so node cannot free the tsfn) until the call returns.
  let registry = CUSTOM_GC_REGISTRY
    .read()
    .unwrap_or_else(|poisoned| poisoned.into_inner());
  if let Some(tsfn) = registry.get(&(env as usize)) {
    let status = unsafe { sys::napi_call_threadsafe_function(tsfn.0, ref_.cast(), 1) };
    // `napi_closing`/`napi_invalid_arg` mean the owning env is tearing down: its
    // references die with the isolate, so the release is moot.
    debug_assert!(
      matches!(
        status,
        sys::Status::napi_ok | sys::Status::napi_closing | sys::Status::napi_invalid_arg
      ),
      "Call custom GC failed {:?}",
      crate::Status::from(status)
    );
  }
  // If absent, the owning environment has been torn down; its references were
  // already freed with the isolate, so there is nothing to release.
}

type RegisteredClasses =
  PersistedPerInstanceHashMap</* export name */ String, /* constructor */ sys::napi_ref>;

#[cfg(feature = "compat-mode")]
// compatibility for #[module_exports]
static MODULE_EXPORTS: Lazy<PersistedPerInstanceVec<ModuleExportsCallback>> =
  Lazy::new(Default::default);

#[inline]
fn wait_first_thread_registered() {
  while !FIRST_MODULE_REGISTERED.load(Ordering::SeqCst) {
    std::hint::spin_loop();
  }
}

#[doc(hidden)]
pub fn get_class_constructor(js_name: &'static str) -> Option<sys::napi_ref> {
  let current_id = std::thread::current().id();
  REGISTERED_CLASSES.borrow_mut(|map| {
    map
      .get(&current_id)
      .map(|m| m.borrow_mut(|map| map.get(js_name).copied()))
  })?
}

#[doc(hidden)]
#[cfg(feature = "compat-mode")]
// compatibility for #[module_exports]
pub fn register_module_exports(callback: ModuleExportsCallback) {
  MODULE_EXPORTS.push(callback);
}

#[doc(hidden)]
pub fn register_module_export(
  js_mod: Option<&'static str>,
  name: &'static str,
  cb: ExportRegisterCallback,
) {
  MODULE_REGISTER_CALLBACK.push((js_mod, (name, cb)));
}

#[doc(hidden)]
pub fn register_js_function(
  name: &'static str,
  cb: ExportRegisterCallback,
  c_fn: sys::napi_callback,
) {
  FN_REGISTER_MAP.borrow_mut(|inner| {
    inner.insert(cb, (c_fn, name));
  });
}

#[doc(hidden)]
pub fn register_class(
  rust_name: &'static str,
  js_mod: Option<&'static str>,
  js_name: &'static str,
  props: Vec<Property>,
) {
  MODULE_CLASS_PROPERTIES.borrow_mut(|inner| {
    let val = inner.entry(rust_name).or_default();
    let val = val.entry(js_mod).or_default();
    val.0 = js_name;
    val.1.extend(props.into_iter());
  });
}

#[inline]
/// Get `JsFunction` from defined Rust `fn`
/// ```rust
/// #[napi]
/// fn some_fn() -> u32 {
///     1
/// }
///
/// #[napi]
/// fn return_some_fn() -> Result<JsFunction> {
///     get_js_function(some_fn_js_function)
/// }
/// ```
///
/// ```js
/// returnSomeFn()(); // 1
/// ```
///
pub fn get_js_function(env: &Env, raw_fn: ExportRegisterCallback) -> Result<JsFunction> {
  FN_REGISTER_MAP.borrow_mut(|inner| {
    inner
      .get(&raw_fn)
      .and_then(|(cb, name)| {
        let mut function = ptr::null_mut();
        let name_len = name.len() - 1;
        let fn_name = unsafe { CStr::from_bytes_with_nul_unchecked(name.as_bytes()) };
        check_status!(unsafe {
          sys::napi_create_function(
            env.0,
            fn_name.as_ptr(),
            name_len,
            *cb,
            ptr::null_mut(),
            &mut function,
          )
        })
        .ok()?;
        Some(JsFunction(Value {
          env: env.0,
          value: function,
          value_type: ValueType::Function,
        }))
      })
      .ok_or_else(|| {
        crate::Error::new(
          crate::Status::InvalidArg,
          "JavaScript function does not exist".to_owned(),
        )
      })
  })
}

/// Get `C Callback` from defined Rust `fn`
/// ```rust
/// #[napi]
/// fn some_fn() -> u32 {
///     1
/// }
///
/// #[napi]
/// fn create_obj(env: Env) -> Result<JsObject> {
///     let mut obj = env.create_object()?;
///     obj.define_property(&[Property::new("getter")?.with_getter(get_c_callback(some_fn_js_function)?)])?;
///     Ok(obj)
/// }
/// ```
///
/// ```js
/// console.log(createObj().getter) // 1
/// ```
///
pub fn get_c_callback(raw_fn: ExportRegisterCallback) -> Result<crate::Callback> {
  FN_REGISTER_MAP.borrow_mut(|inner| {
    inner
      .get(&raw_fn)
      .and_then(|(cb, _name)| *cb)
      .ok_or_else(|| {
        crate::Error::new(
          crate::Status::InvalidArg,
          "JavaScript function does not exist".to_owned(),
        )
      })
  })
}

#[cfg(windows)]
#[ctor::ctor]
fn load_host() {
  unsafe {
    sys::setup();
  }
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
unsafe extern "C" fn napi_register_wasm_v1(
  env: sys::napi_env,
  exports: sys::napi_value,
) -> sys::napi_value {
  unsafe { napi_register_module_v1(env, exports) }
}

#[no_mangle]
unsafe extern "C" fn napi_register_module_v1(
  env: sys::napi_env,
  exports: sys::napi_value,
) -> sys::napi_value {
  if IS_FIRST_MODULE.load(Ordering::SeqCst) {
    IS_FIRST_MODULE.store(false, Ordering::SeqCst);
  } else {
    wait_first_thread_registered();
  }
  let mut exports_objects: HashSet<String> = HashSet::default();
  MODULE_REGISTER_CALLBACK.borrow_mut(|inner| {
    inner
      .iter_mut()
      .fold(
        HashMap::<Option<&'static str>, Vec<(&'static str, ExportRegisterCallback)>>::new(),
        |mut acc, (js_mod, item)| {
          if let Some(k) = acc.get_mut(js_mod) {
            k.push(*item);
          } else {
            acc.insert(*js_mod, vec![*item]);
          }
          acc
        },
      )
      .iter()
      .for_each(|(js_mod, items)| {
        let mut exports_js_mod = ptr::null_mut();
        if let Some(js_mod_str) = js_mod {
          let mod_name_c_str =
            unsafe { CStr::from_bytes_with_nul_unchecked(js_mod_str.as_bytes()) };
          if exports_objects.contains(*js_mod_str) {
            check_status_or_throw!(
              env,
              unsafe {
                sys::napi_get_named_property(
                  env,
                  exports,
                  mod_name_c_str.as_ptr(),
                  &mut exports_js_mod,
                )
              },
              "Get mod {} from exports failed",
              js_mod_str,
            );
          } else {
            check_status_or_throw!(
              env,
              unsafe { sys::napi_create_object(env, &mut exports_js_mod) },
              "Create export JavaScript Object [{}] failed",
              js_mod_str
            );
            check_status_or_throw!(
              env,
              unsafe {
                sys::napi_set_named_property(env, exports, mod_name_c_str.as_ptr(), exports_js_mod)
              },
              "Set exports Object [{}] into exports object failed",
              js_mod_str
            );
            exports_objects.insert(js_mod_str.to_string());
          }
        }
        for (name, callback) in items {
          unsafe {
            let js_name = CStr::from_bytes_with_nul_unchecked(name.as_bytes());
            if let Err(e) = callback(env).and_then(|v| {
              let exported_object = if exports_js_mod.is_null() {
                exports
              } else {
                exports_js_mod
              };
              check_status!(
                sys::napi_set_named_property(env, exported_object, js_name.as_ptr(), v),
                "Failed to register export `{}`",
                name,
              )
            }) {
              JsError::from(e).throw_into(env)
            }
          }
        }
      })
  });

  let mut registered_classes = HashMap::new();

  MODULE_CLASS_PROPERTIES.borrow_mut(|inner| {
    inner.iter().for_each(|(rust_name, js_mods)| {
      for (js_mod, (js_name, props)) in js_mods {
        let mut exports_js_mod = ptr::null_mut();
        unsafe {
          if let Some(js_mod_str) = js_mod {
            let mod_name_c_str = CStr::from_bytes_with_nul_unchecked(js_mod_str.as_bytes());
            if exports_objects.contains(*js_mod_str) {
              check_status_or_throw!(
                env,
                sys::napi_get_named_property(
                  env,
                  exports,
                  mod_name_c_str.as_ptr(),
                  &mut exports_js_mod,
                ),
                "Get mod {} from exports failed",
                js_mod_str,
              );
            } else {
              check_status_or_throw!(
                env,
                sys::napi_create_object(env, &mut exports_js_mod),
                "Create export JavaScript Object [{}] failed",
                js_mod_str
              );
              check_status_or_throw!(
                env,
                sys::napi_set_named_property(env, exports, mod_name_c_str.as_ptr(), exports_js_mod),
                "Set exports Object [{}] into exports object failed",
                js_mod_str
              );
              exports_objects.insert(js_mod_str.to_string());
            }
          }
          let (ctor, props): (Vec<_>, Vec<_>) = props.iter().partition(|prop| prop.is_ctor);

          let ctor = ctor.get(0).map(|c| c.raw().method.unwrap()).unwrap_or(noop);
          let raw_props: Vec<_> = props.iter().map(|prop| prop.raw()).collect();

          let js_class_name = CStr::from_bytes_with_nul_unchecked(js_name.as_bytes());
          let mut class_ptr = ptr::null_mut();

          check_status_or_throw!(
            env,
            sys::napi_define_class(
              env,
              js_class_name.as_ptr(),
              js_name.len() - 1,
              Some(ctor),
              ptr::null_mut(),
              raw_props.len(),
              raw_props.as_ptr(),
              &mut class_ptr,
            ),
            "Failed to register class `{}` generate by struct `{}`",
            &js_name,
            &rust_name
          );

          let mut ctor_ref = ptr::null_mut();
          sys::napi_create_reference(env, class_ptr, 1, &mut ctor_ref);

          registered_classes.insert(js_name.to_string(), ctor_ref);

          check_status_or_throw!(
            env,
            sys::napi_set_named_property(
              env,
              if exports_js_mod.is_null() {
                exports
              } else {
                exports_js_mod
              },
              js_class_name.as_ptr(),
              class_ptr
            ),
            "Failed to register class `{}` generate by struct `{}`",
            &js_name,
            &rust_name
          );
        }
      }
    });

    REGISTERED_CLASSES.borrow_mut(|map| {
      map.insert(
        std::thread::current().id(),
        PersistedPerInstanceHashMap::from_hashmap(registered_classes),
      )
    });
  });

  #[cfg(feature = "compat-mode")]
  MODULE_EXPORTS.borrow_mut(|inner| {
    inner.iter().for_each(|callback| unsafe {
      if let Err(e) = callback(env, exports) {
        JsError::from(e).throw_into(env);
      }
    })
  });

  #[cfg(all(windows, feature = "napi4", feature = "tokio_rt"))]
  {
    crate::tokio_runtime::ensure_runtime();

    crate::tokio_runtime::RT_REFERENCE_COUNT.fetch_add(1, Ordering::SeqCst);
    unsafe {
      sys::napi_add_env_cleanup_hook(
        env,
        Some(crate::tokio_runtime::drop_runtime),
        ptr::null_mut(),
      )
    };
  }
  #[cfg(all(feature = "napi4", not(target_arch = "wasm32")))]
  create_custom_gc(env);
  FIRST_MODULE_REGISTERED.store(true, Ordering::SeqCst);
  exports
}

pub(crate) unsafe extern "C" fn noop(
  env: sys::napi_env,
  _info: sys::napi_callback_info,
) -> sys::napi_value {
  if !crate::bindgen_runtime::___CALL_FROM_FACTORY.with(|s| s.load(Ordering::Relaxed)) {
    unsafe {
      sys::napi_throw_error(
        env,
        ptr::null_mut(),
        CStr::from_bytes_with_nul_unchecked(b"Class contains no `constructor`, can not new it!\0")
          .as_ptr(),
      );
    }
  }
  ptr::null_mut()
}

#[cfg(all(feature = "napi4", not(target_arch = "wasm32")))]
fn create_custom_gc(env: sys::napi_env) {
  use std::os::raw::c_char;

  let mut custom_gc_fn = ptr::null_mut();
  check_status_or_throw!(
    env,
    unsafe {
      sys::napi_create_function(
        env,
        "custom_gc".as_ptr() as *const c_char,
        9,
        Some(empty),
        ptr::null_mut(),
        &mut custom_gc_fn,
      )
    },
    "Create Custom GC Function in napi_register_module_v1 failed"
  );
  let mut async_resource_name = ptr::null_mut();
  check_status_or_throw!(
    env,
    unsafe {
      sys::napi_create_string_utf8(
        env,
        "CustomGC".as_ptr() as *const c_char,
        8,
        &mut async_resource_name,
      )
    },
    "Create async resource string in napi_register_module_v1"
  );
  let mut custom_gc_tsfn = ptr::null_mut();
  check_status_or_throw!(
    env,
    unsafe {
      sys::napi_create_threadsafe_function(
        env,
        custom_gc_fn,
        ptr::null_mut(),
        async_resource_name,
        0,
        1,
        ptr::null_mut(),
        Some(custom_gc_finalize),
        ptr::null_mut(),
        Some(custom_gc),
        &mut custom_gc_tsfn,
      )
    },
    "Create Custom GC ThreadsafeFunction in napi_register_module_v1 failed"
  );
  check_status_or_throw!(
    env,
    unsafe { sys::napi_unref_threadsafe_function(env, custom_gc_tsfn) },
    "Unref Custom GC ThreadsafeFunction in napi_register_module_v1 failed"
  );
  // Register this environment's tsfn keyed by `env`, and remember which env this
  // thread belongs to so off-thread drops can be routed back to it.
  CUSTOM_GC_REGISTRY
    .write()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
    .insert(env as usize, CustomGcTsfn(custom_gc_tsfn));
  CURRENT_ENV.with(|cell| cell.set(env));
}

#[cfg(all(feature = "napi4", not(target_arch = "wasm32")))]
#[allow(unused)]
unsafe extern "C" fn empty(env: sys::napi_env, info: sys::napi_callback_info) -> sys::napi_value {
  ptr::null_mut()
}

#[cfg(all(feature = "napi4", not(target_arch = "wasm32")))]
#[allow(unused)]
unsafe extern "C" fn custom_gc_finalize(
  env: sys::napi_env,
  finalize_data: *mut std::ffi::c_void,
  finalize_hint: *mut std::ffi::c_void,
) {
  // This environment is being torn down; remove its tsfn from the registry so no
  // further off-thread release is routed to a dead isolate. The write lock
  // serializes against `route_custom_gc_unref` (holding the read lock while
  // calling the tsfn), so the tsfn is never freed mid-call.
  CUSTOM_GC_REGISTRY
    .write()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
    .remove(&(env as usize));
}

#[cfg(all(feature = "napi4", not(target_arch = "wasm32")))]
// recycle the ArrayBuffer/Buffer Reference if the ArrayBuffer/Buffer is not dropped on the main thread
extern "C" fn custom_gc(
  env: sys::napi_env,
  _js_callback: sys::napi_value,
  _context: *mut std::ffi::c_void,
  data: *mut std::ffi::c_void,
) {
  if data.is_null() {
    return;
  }
  // Runs on the owning environment's own thread (this tsfn belongs to `env`).
  // Best-effort: never throw from here.
  unsafe { sys::napi_delete_reference(env, data as sys::napi_ref) };
}
