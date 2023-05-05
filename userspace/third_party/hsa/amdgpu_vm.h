#pragma once

#define AMDGPU_PTE_VALID	(1ULL << 0)
#define AMDGPU_PTE_SYSTEM	(1ULL << 1)
#define AMDGPU_PTE_SNOOPED	(1ULL << 2)

/* RV+ */
#define AMDGPU_PTE_TMZ		(1ULL << 3)

/* VI only */
#define AMDGPU_PTE_EXECUTABLE	(1ULL << 4)

#define AMDGPU_PTE_READABLE	(1ULL << 5)
#define AMDGPU_PTE_WRITEABLE	(1ULL << 6)

#define AMDGPU_PTE_FRAG(x)	((x & 0x1fULL) << 7)

/* TILED for VEGA10, reserved for older ASICs  */
#define AMDGPU_PTE_PRT		(1ULL << 51)

/* PDE is handled as PTE for VEGA10 */
#define AMDGPU_PDE_PTE		(1ULL << 54)

#define AMDGPU_PTE_LOG          (1ULL << 55)

/* PTE is handled as PDE for VEGA10 (Translate Further) */
#define AMDGPU_PTE_TF		(1ULL << 56)

/* MALL noalloc for sienna_cichlid, reserved for older ASICs  */
#define AMDGPU_PTE_NOALLOC	(1ULL << 58)

/* PDE Block Fragment Size for VEGA10 */
#define AMDGPU_PDE_BFS(a)	((uint64_t)a << 59)