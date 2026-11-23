#include "cframetap.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

void *sp_shm_map(const char *path, size_t size, int *err) {
    int fd = shm_open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        *err = errno;
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_size == 0) {
        if (ftruncate(fd, (off_t)size) != 0 && errno != EINVAL) {  // EINVAL: the other side sized it first
            *err = errno;
            close(fd);
            return NULL;
        }
    }
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < size) {  // (macOS rounds the size up to whole pages)
        *err = -1;
        close(fd);
        return NULL;
    }
    void *base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    int e = errno;
    close(fd);
    if (base == MAP_FAILED) {
        *err = e;
        return NULL;
    }
    *err = 0;
    return base;
}

void sp_shm_unmap(void *base, size_t size) {
    if (base) munmap(base, size);
}

int sp_shm_unlink(const char *path) { return shm_unlink(path) == 0 ? 0 : errno; }

void sp_fence(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }

void sp_store_u64(void *at, uint64_t v) { __atomic_store_n((uint64_t *)at, v, __ATOMIC_SEQ_CST); }

uint64_t sp_load_u64(const void *at) { return __atomic_load_n((const uint64_t *)at, __ATOMIC_SEQ_CST); }

void sp_store_u32(void *at, uint32_t v) { __atomic_store_n((uint32_t *)at, v, __ATOMIC_SEQ_CST); }

uint32_t sp_load_u32(const void *at) { return __atomic_load_n((const uint32_t *)at, __ATOMIC_SEQ_CST); }

uint64_t sp_now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
