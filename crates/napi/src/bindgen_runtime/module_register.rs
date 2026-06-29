#[cfg(not(feature = "noop"))]
use std::cell::Cell;
use std::cell::{LazyCell, RefCell};
#[cfg(not(feature = "noop"))]
use std::collections::HashSet;
#[cfg(not(feature = "noop"))]
use std::ffi::CStr;
#[cfg(all(not(feature = "noop"), feature = "node_version_detect"))]
use std::mem::MaybeUninit;
#[cfg(not(feature = "noop"))]
use std::ptr;
#[cfg(all(not(feature = "noop"), feature = "node_version_detect"))]
use std::sync::OnceLock;
#[cfg(not(feature = "noop"))]
use std::sync::{
  atomic::{AtomicBool, AtomicUsize, Ordering},
  LazyLock, RwLock,
};
use std::{any::TypeId, collections::HashMap};

use rustc_hash::FxBuildHasher;

#[cfg(all(not(feature = "noop"), feature = "node_version_detect"))]
use crate::NodeVersion;
#[cfg(not(feature = "noop"))]
use crate::{check_status, check_status_or_throw, JsError};
use crate::{sys, Property, Result};

// #[napi] fn
pub type ExportRegisterCallback = unsafe fn(sys::napi_env) -> Result<sys::napi_value>;
// #[napi(module_exports)] fn
pub type ExportRegisterHookCallback =
  unsafe fn(sys::napi_env, sys::napi_value) -> Result<sys::napi_value>;
pub type ModuleExportsCallback =
  unsafe fn(env: sys::napi_env, exports: sys::napi_value) -> Result<()>;

#[cfg(all(not(feature = "noop"), feature = "node_version_detect"))]
pub static NODE_VERSION: OnceLock<NodeVersion> = OnceLock::new();

#[cfg(feature = "node_version_detect")]
pub static mut NODE_VERSION_MAJOR: u32 = 0;
#[cfg(feature = "node_version_detect")]
pub static mut NODE_VERSION_MINOR: u32 = 0;
#[cfg(feature = "node_version_detect")]
pub static mut NODE_VERSION_PATCH: u32 = 0;

#[repr(transparent)]
pub(crate) struct PersistedPerInstanceHashMap<K, V, S>(RefCell<HashMap<K, V, S>>);

impl<K, V, S> PersistedPerInstanceHashMap<K, V, S> {
  #[allow(clippy::mut_from_ref)]
  pub(crate) fn borrow_mut<F, R>(&self, f: F) -> R
  where
    F: FnOnce(&mut HashMap<K, V, S>) -> R,
  {
    f(&mut *self.0.borrow_mut())
  }
}

impl<K, V, S: Default> Default for PersistedPerInstanceHashMap<K, V, S> {
  fn default() -> Self {
    Self(RefCell::new(HashMap::<K, V, S>::default()))
  }
}

#[cfg(not(feature = "noop"))]
type ModuleRegisterCallback =
  RwLock<Vec<(Option<&'static str>, (&'static str, ExportRegisterCallback))>>;

#[cfg(not(feature = "noop"))]
type ClassPropertyRegistry =
  HashMap<TypeId, HashMap<Option<&'static str>, ClassRegistration, FxBuildHasher>, FxBuildHasher>;

#[cfg(not(feature = "noop"))]
struct ClassRegistration {
  js_name: &'static str,
  props: Vec<Property>,
  implement_iterator: bool,
}

// Stores class metadata registered by napi macros.
// Since class properties do not contain any napi_value, ModuleClassProperty is thread-safe.
// This structure is shared between the main JS thread and worker threads.
#[cfg(not(feature = "noop"))]
#[derive(Default)]
struct ModuleClassProperty(RwLock<ClassPropertyRegistry>);

#[cfg(not(feature = "noop"))]
unsafe impl Send for ModuleClassProperty {}
#[cfg(not(feature = "noop"))]
unsafe impl Sync for ModuleClassProperty {}

#[cfg(not(feature = "noop"))]
impl ModuleClassProperty {
  pub(crate) fn borrow_mut<F, R>(&self, f: F) -> R
  where
    F: FnOnce(&mut ClassPropertyRegistry) -> R,
  {
    let mut write_lock = self.0.write().unwrap();
    f(&mut write_lock)
  }

  pub(crate) fn borrow<F, R>(&self, f: F) -> R
  where
    F: FnOnce(&ClassPropertyRegistry) -> R,
  {
    let write_lock = self.0.read().unwrap();
    f(&write_lock)
  }
}

#[cfg(not(feature = "noop"))]
static MODULE_REGISTER_CALLBACK: LazyLock<ModuleRegisterCallback> = LazyLock::new(Default::default);
#[cfg(not(feature = "noop"))]
static MODULE_REGISTER_HOOK_CALLBACK: LazyLock<RwLock<Option<ExportRegisterHookCallback>>> =
  LazyLock::new(Default::default);
#[cfg(not(feature = "noop"))]
static MODULE_CLASS_PROPERTIES: LazyLock<ModuleClassProperty> = LazyLock::new(Default::default);
#[cfg(not(feature = "noop"))]
static MODULE_COUNT: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(feature = "noop"))]
static FIRST_MODULE_REGISTERED: AtomicBool = AtomicBool::new(false);
#[cfg(all(
  feature = "tokio_rt",
  not(target_family = "wasm"),
  not(feature = "noop")
))]
static ENV_CLEANUP_HOOK_ADDED: RwLock<bool> = RwLock::new(false);
thread_local! {
  static REGISTERED_CLASSES: LazyCell<RegisteredClasses> = LazyCell::new(Default::default);
}
// Per-environment CustomGC registry.
//
// A `Buffer`/`TypedArray` received from JS carries a `napi_reference` that is
// bound to the V8 isolate of the environment it was created in. That reference
// can only be released on *that* environment's thread — V8 global handles are
// isolate-bound, and touching one from another isolate's thread is a fatal
// error (`Check failed: node->IsInUse()` in `GlobalHandles::MakeWeak`).
//
// When such a value is dropped on a non-owning thread (e.g. it was moved into an
// `async fn` future running on a tokio worker, or sent to another thread), we
// route its unref back to the owning environment's thread through a CustomGC
// `ThreadsafeFunction`. Each environment (the main thread and every
// `worker_thread`) registers its OWN tsfn here, keyed by its `napi_env` pointer,
// so the unref always lands on the correct isolate.
//
// (Previously a single process-global tsfn was shared by all environments, so an
// off-thread unref of a worker's Buffer was routed to whichever environment
// loaded the addon first — the wrong, or an already-torn-down, isolate.)
#[cfg(all(feature = "napi4", not(feature = "noop")))]
pub(crate) struct CustomGcTsfn(pub(crate) sys::napi_threadsafe_function);
// SAFETY: a `napi_threadsafe_function` is explicitly designed to be invoked from
// any thread (`napi_call_threadsafe_function`).
#[cfg(all(feature = "napi4", not(feature = "noop")))]
unsafe impl Send for CustomGcTsfn {}
#[cfg(all(feature = "napi4", not(feature = "noop")))]
unsafe impl Sync for CustomGcTsfn {}

#[cfg(all(feature = "napi4", not(feature = "noop")))]
pub(crate) static CUSTOM_GC_REGISTRY: LazyLock<RwLock<HashMap<usize, CustomGcTsfn>>> =
  LazyLock::new(|| RwLock::new(HashMap::default()));

thread_local! {
  #[cfg(all(feature = "napi4", not(feature = "noop")))]
  // The `napi_env` of the environment this thread belongs to, set when the
  // module registers on this thread. Null on non-env threads (tokio workers,
  // user-spawned threads). Used to decide whether a Buffer/TypedArray reference
  // can be released directly (we are on its owning env thread) or must be routed
  // through that env's CustomGC tsfn.
  pub(crate) static CURRENT_ENV: Cell<sys::napi_env> = const { Cell::new(ptr::null_mut()) };
  #[cfg(all(feature = "napi4", not(feature = "noop")))]
  // Whether the current thread is an environment thread that can touch napi
  // references at all. Used to guard the CustomGC callback.
  pub(crate) static THREADS_CAN_ACCESS_ENV: Cell<bool> = const { Cell::new(false) };
}

/// Release a `napi_reference` held by a dropping `Buffer`/`TypedArray`.
///
/// `env` is the environment the reference was created in. If we are on that
/// environment's own thread the reference is released directly; otherwise the
/// release is routed to the owning env's thread through its CustomGC tsfn so the
/// `napi_reference_unref` runs on the correct isolate.
#[cfg(all(feature = "napi4", not(feature = "noop")))]
pub(crate) fn route_custom_gc_unref(ref_: sys::napi_ref, env: sys::napi_env) {
  if ref_.is_null() || env.is_null() {
    return;
  }
  // On the owning environment's own thread: release directly.
  if CURRENT_ENV.with(|cell| cell.get()) == env {
    unsafe { release_buffer_ref(env, ref_) };
    return;
  }
  // Off the owning thread: route the unref to the owning env's CustomGC tsfn so
  // it runs on the correct isolate. Holding the read lock across the call keeps
  // the tsfn alive against concurrent env teardown — `custom_gc_finalize` takes
  // the write lock (so node cannot free the tsfn) until the call returns.
  let registry = CUSTOM_GC_REGISTRY
    .read()
    .unwrap_or_else(|poisoned| poisoned.into_inner());
  if let Some(tsfn) = registry.get(&(env as usize)) {
    let status = unsafe {
      sys::napi_call_threadsafe_function(
        tsfn.0,
        ref_.cast(),
        sys::ThreadsafeFunctionCallMode::blocking,
      )
    };
    // `napi_closing`/`napi_invalid_arg` mean the owning env is tearing down: its
    // references die with the isolate, so the unref is moot. Anything else is
    // unexpected and worth catching in debug builds.
    debug_assert!(
      matches!(
        status,
        sys::Status::napi_ok | sys::Status::napi_closing | sys::Status::napi_invalid_arg
      ),
      "Call custom GC failed {}",
      crate::Status::from(status)
    );
  }
  // If absent, the owning environment has been torn down; its references were
  // already freed with the isolate, so there is nothing to release.
}

/// Release a buffer/typedarray `napi_reference` on its owning environment's
/// thread. Best-effort: a dropping value must never throw, and during env
/// teardown these calls can fail (the isolate frees the handle regardless).
#[cfg(all(feature = "napi4", not(feature = "noop")))]
unsafe fn release_buffer_ref(env: sys::napi_env, ref_: sys::napi_ref) {
  let mut ref_count = 0;
  let status = unsafe { sys::napi_reference_unref(env, ref_, &mut ref_count) };
  debug_assert!(
    status != sys::Status::napi_ok || ref_count == 0,
    "Buffer reference count in drop is not zero"
  );
  unsafe { sys::napi_delete_reference(env, ref_) };
}

type RegisteredClasses = PersistedPerInstanceHashMap<
  /* export name */ String,
  /* constructor */ sys::napi_ref,
  FxBuildHasher,
>;

#[cfg(all(feature = "compat-mode", not(feature = "noop")))]
// compatibility for #[module_exports]
static MODULE_EXPORTS: LazyLock<RwLock<Vec<ModuleExportsCallback>>> =
  LazyLock::new(Default::default);

#[cfg(not(feature = "noop"))]
#[inline]
fn wait_first_thread_registered() {
  while !FIRST_MODULE_REGISTERED.load(Ordering::SeqCst) {
    std::hint::spin_loop();
  }
}

#[doc(hidden)]
#[cfg(all(feature = "compat-mode", not(feature = "noop")))]
// compatibility for #[module_exports]
pub fn register_module_exports(callback: ModuleExportsCallback) {
  MODULE_EXPORTS
    .write()
    .expect("Register module exports failed")
    .push(callback);
}

#[cfg(feature = "noop")]
#[doc(hidden)]
pub fn register_module_exports(_: ModuleExportsCallback) {}

#[cfg(not(feature = "noop"))]
#[doc(hidden)]
pub fn register_module_export(
  js_mod: Option<&'static str>,
  name: &'static str,
  cb: ExportRegisterCallback,
) {
  MODULE_REGISTER_CALLBACK
    .write()
    .expect("Register module export failed")
    .push((js_mod, (name, cb)));
}

#[cfg(feature = "noop")]
#[doc(hidden)]
pub fn register_module_export(
  _js_mod: Option<&'static str>,
  _name: &'static str,
  _cb: ExportRegisterCallback,
) {
}

#[cfg(not(feature = "noop"))]
#[doc(hidden)]
pub fn register_module_export_hook(cb: ExportRegisterHookCallback) {
  let mut inner = MODULE_REGISTER_HOOK_CALLBACK
    .write()
    .expect("Write MODULE_REGISTER_HOOK_CALLBACK failed");
  *inner = Some(cb);
}

#[cfg(feature = "noop")]
#[doc(hidden)]
pub fn register_module_export_hook(_cb: ExportRegisterHookCallback) {}

#[doc(hidden)]
pub fn get_class_constructor(js_name: &'static str) -> Option<sys::napi_ref> {
  REGISTERED_CLASSES.with(|cell| cell.borrow_mut(|map| map.get(js_name).copied()))
}

#[cfg(not(feature = "noop"))]
#[doc(hidden)]
pub fn register_class(
  rust_type_id: TypeId,
  js_mod: Option<&'static str>,
  js_name: &'static str,
  props: Vec<Property>,
  implement_iterator: bool,
) {
  MODULE_CLASS_PROPERTIES.borrow_mut(|inner| {
    let val = inner.entry(rust_type_id).or_default();
    let val = val.entry(js_mod).or_insert_with(|| ClassRegistration {
      js_name,
      props: Vec::new(),
      implement_iterator,
    });
    val.js_name = js_name;
    val.implement_iterator |= implement_iterator;
    val.props.extend(props);
  });
}

#[cfg(feature = "noop")]
#[doc(hidden)]
#[allow(unused_variables)]
pub fn register_class(
  rust_type_id: TypeId,
  js_mod: Option<&'static str>,
  js_name: &'static str,
  props: Vec<Property>,
  implement_iterator: bool,
) {
}

#[cfg(all(target_family = "wasm", not(feature = "noop")))]
#[no_mangle]
unsafe extern "C" fn napi_register_wasm_v1(
  env: sys::napi_env,
  exports: sys::napi_value,
) -> sys::napi_value {
  unsafe { napi_register_module_v1(env, exports) }
}

#[cfg(not(feature = "noop"))]
#[no_mangle]
/// Register the n-api module exports.
///
/// # Safety
/// This method is meant to be called by Node.js while importing the n-api module.
/// Only call this method if the current module is **not** imported by a node-like runtime.
///
/// Arguments `env` and `exports` must **not** be null.
pub unsafe extern "C" fn napi_register_module_v1(
  env: sys::napi_env,
  exports: sys::napi_value,
) -> sys::napi_value {
  #[cfg(any(
    target_env = "msvc",
    all(not(target_family = "wasm"), feature = "dyn-symbols")
  ))]
  unsafe {
    sys::setup();
  }
  #[cfg(feature = "node_version_detect")]
  {
    NODE_VERSION.get_or_init(|| {
      let mut node_version = MaybeUninit::uninit();
      check_status_or_throw!(
        env,
        unsafe { sys::napi_get_node_version(env, node_version.as_mut_ptr()) },
        "Failed to get node version"
      );
      let node_version = *node_version.assume_init();
      unsafe {
        NODE_VERSION_MAJOR = node_version.major;
        NODE_VERSION_MINOR = node_version.minor;
        NODE_VERSION_PATCH = node_version.patch;
      }
      NodeVersion {
        major: node_version.major,
        minor: node_version.minor,
        patch: node_version.patch,
        release: unsafe { CStr::from_ptr(node_version.release).to_str().unwrap() },
      }
    });
  }

  if MODULE_COUNT.fetch_add(1, Ordering::SeqCst) != 0 {
    wait_first_thread_registered();
  }

  let mut exports_objects: HashSet<String> = HashSet::default();

  {
    let mut register_callback = MODULE_REGISTER_CALLBACK
      .write()
      .expect("Write MODULE_REGISTER_CALLBACK in napi_register_module_v1 failed");
    register_callback
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
      });
  }

  let mut registered_classes = HashMap::default();

  MODULE_CLASS_PROPERTIES.borrow(|inner| {
    inner.iter().for_each(|(_, js_mods)| {
      for (js_mod, class_registration) in js_mods {
        let mut exports_js_mod = ptr::null_mut();
        unsafe {
          let js_name = class_registration.js_name;
          let props = &class_registration.props;
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

          let ctor = ctor
            .first()
            .map(|c| c.raw().method.unwrap())
            .unwrap_or(noop);
          let raw_props: Vec<_> = props.iter().map(|prop| prop.raw()).collect();

          let js_class_name = CStr::from_bytes_with_nul_unchecked(js_name.as_bytes());
          let mut class_ptr = ptr::null_mut();

          check_status_or_throw!(
            env,
            sys::napi_define_class(
              env,
              js_class_name.as_ptr(),
              js_name.len() as isize - 1,
              Some(ctor),
              ptr::null_mut(),
              raw_props.len(),
              raw_props.as_ptr(),
              &mut class_ptr,
            ),
            "Failed to register class `{}`",
            &js_name,
          );

          if class_registration.implement_iterator {
            crate::bindgen_runtime::iterator::setup_iterator_class(env, class_ptr);
          }

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
            "Failed to register class `{}`",
            &js_name,
          );
        }
      }
    });
  });

  REGISTERED_CLASSES.with(|cell| {
    cell.borrow_mut(|map| {
      *map = registered_classes;
    })
  });

  let module_register_hook_callback = MODULE_REGISTER_HOOK_CALLBACK
    .read()
    .expect("Read MODULE_REGISTER_HOOK_CALLBACK failed");
  if let Some(cb) = module_register_hook_callback.as_ref() {
    if let Err(e) = cb(env, exports) {
      JsError::from(e).throw_into(env);
    }
  }

  #[cfg(feature = "compat-mode")]
  {
    let module_exports = MODULE_EXPORTS.read().expect("Read MODULE_EXPORTS failed");
    module_exports.iter().for_each(|callback| unsafe {
      if let Err(e) = callback(env, exports) {
        JsError::from(e).throw_into(env);
      }
    })
  }

  #[cfg(feature = "napi4")]
  {
    create_custom_gc(env);
    #[cfg(feature = "tokio_rt")]
    {
      crate::tokio_runtime::start_async_runtime();
      #[cfg(not(target_family = "wasm"))]
      {
        let mut env_cleanup_hook_added = ENV_CLEANUP_HOOK_ADDED.write().unwrap();
        if !*env_cleanup_hook_added {
          check_status_or_throw!(
            env,
            unsafe { sys::napi_add_env_cleanup_hook(env, Some(thread_cleanup), ptr::null_mut()) },
            "Failed to add env cleanup hook"
          );
          *env_cleanup_hook_added = true;
          drop(env_cleanup_hook_added);
        }
      }
    }
  }

  #[cfg(all(feature = "tokio_rt", feature = "napi4", target_family = "wasm"))]
  check_status_or_throw!(
    env,
    unsafe {
      sys::napi_wrap(
        env,
        exports,
        std::ptr::null_mut(),
        Some(thread_cleanup),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
      )
    },
    "Failed to add remove thread id cleanup hook"
  );

  FIRST_MODULE_REGISTERED.store(true, Ordering::SeqCst);
  exports
}

#[cfg(not(feature = "noop"))]
pub(crate) unsafe extern "C" fn noop(
  env: sys::napi_env,
  _info: sys::napi_callback_info,
) -> sys::napi_value {
  if !crate::bindgen_runtime::___CALL_FROM_FACTORY.with(|s| s.get()) {
    unsafe {
      sys::napi_throw_error(
        env,
        ptr::null_mut(),
        c"Class contains no `constructor`, can not new it!".as_ptr(),
      );
    }
  }
  ptr::null_mut()
}

#[cfg(all(feature = "napi4", not(feature = "noop")))]
fn create_custom_gc(env: sys::napi_env) {
  // Create a CustomGC ThreadsafeFunction *per environment* (main thread and each
  // worker_thread), keyed by `env` in `CUSTOM_GC_REGISTRY`. Off-thread Buffer /
  // TypedArray unrefs are routed back to their owning env's tsfn so the
  // `napi_reference_unref` always runs on the correct isolate.
  let mut custom_gc_fn = ptr::null_mut();
  check_status_or_throw!(
    env,
    unsafe {
      sys::napi_create_function(
        env,
        c"custom_gc".as_ptr(),
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
    unsafe { sys::napi_create_string_utf8(env, c"CustomGC".as_ptr(), 8, &mut async_resource_name) },
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
  CUSTOM_GC_REGISTRY
    .write()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
    .insert(env as usize, CustomGcTsfn(custom_gc_tsfn));

  CURRENT_ENV.with(|cell| cell.set(env));
  THREADS_CAN_ACCESS_ENV.with(|cell| cell.set(true));
}

#[cfg(all(
  not(feature = "noop"),
  all(feature = "tokio_rt", feature = "napi4"),
  not(target_family = "wasm")
))]
unsafe extern "C" fn thread_cleanup(_data: *mut std::ffi::c_void) {
  if MODULE_COUNT.fetch_sub(1, Ordering::Relaxed) == 1 {
    crate::tokio_runtime::shutdown_async_runtime();
  }
}

#[cfg(all(
  not(feature = "noop"),
  all(feature = "tokio_rt", feature = "napi4"),
  target_family = "wasm"
))]
unsafe extern "C" fn thread_cleanup(
  _env: sys::napi_env,
  _id: *mut std::ffi::c_void,
  _data: *mut std::ffi::c_void,
) {
  if MODULE_COUNT.fetch_sub(1, Ordering::Relaxed) == 1 {
    crate::tokio_runtime::shutdown_async_runtime();
  }
}

#[cfg(all(feature = "napi4", not(feature = "noop")))]
#[allow(unused)]
unsafe extern "C" fn empty(env: sys::napi_env, info: sys::napi_callback_info) -> sys::napi_value {
  ptr::null_mut()
}

#[cfg(all(feature = "napi4", not(feature = "noop")))]
#[allow(unused_variables)]
unsafe extern "C" fn custom_gc_finalize(
  env: sys::napi_env,
  finalize_data: *mut std::ffi::c_void,
  finalize_hint: *mut std::ffi::c_void,
) {
  // This environment is being torn down; remove its tsfn from the registry so no
  // further off-thread unref is routed to a dead isolate. Taking the write lock
  // here serializes against `route_custom_gc_unref` (which holds the read lock
  // while calling the tsfn), so the tsfn is never freed mid-call.
  CUSTOM_GC_REGISTRY
    .write()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
    .remove(&(env as usize));
}

#[cfg(all(feature = "napi4", not(feature = "noop")))]
// recycle the ArrayBuffer/Buffer Reference if the ArrayBuffer/Buffer is not dropped on the main thread
extern "C" fn custom_gc(
  env: sys::napi_env,
  _js_callback: sys::napi_value,
  _context: *mut std::ffi::c_void,
  data: *mut std::ffi::c_void,
) {
  // current thread was destroyed
  if THREADS_CAN_ACCESS_ENV.with(|cell| !cell.get()) || data.is_null() {
    return;
  }
  // Runs on the owning environment's own thread (this tsfn belongs to `env`), so
  // the unref always targets the correct isolate. Best-effort: never throw from
  // here.
  unsafe { release_buffer_ref(env, data.cast()) };
}
