// The frame tap's shared memory, for the Swift writer (FrameTap.swift). The layout and the protocol are
// docs/frame-tap.md's; these are only the calls Swift can't make itself.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opens (creating it if needed) the POSIX shared memory `path` ("/smoovplay-window"), sizes it to `size` bytes if
/// it's still empty (macOS sizes a shared-memory object only once; EINVAL from a second try is fine) and maps it.
/// Returns the mapping, or NULL with *err set to errno (or to -1 if the object exists but is smaller than `size`).
void *sp_shm_map(const char *path, size_t size, int *err);

/// Unmaps what sp_shm_map returned.
void sp_shm_unmap(void *base, size_t size);

/// Removes the shared memory `path` (tests only: the tap is meant to last until a restart). 0, or errno.
int sp_shm_unlink(const char *path);

/// A full barrier (Apple's chips may show another core stores out of order).
void sp_fence(void);

/// Stores and loads with the barrier the protocol needs.
void sp_store_u64(void *at, uint64_t v);
uint64_t sp_load_u64(const void *at);
void sp_store_u32(void *at, uint32_t v);
uint32_t sp_load_u32(const void *at);

/// Now, in nanoseconds of CLOCK_UPTIME_RAW: what the reader (Python's time.monotonic_ns()) compares with.
uint64_t sp_now_ns(void);

#ifdef __cplusplus
}
#endif
