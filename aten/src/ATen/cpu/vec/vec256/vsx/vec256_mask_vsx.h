#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec_mask.h>

namespace at::vec {
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_VSX)

// safe_bitcast: avoid strict-aliasing by using memcpy
template <typename To, typename From>
static inline To safe_bitcast(const From &from) {
  static_assert(sizeof(To) == sizeof(From), "safe_bitcast size mismatch");
  To to;
  std::memcpy(&to, &from, sizeof(to));
  return to;
}

template <int dst_n, typename mask_t, int mask_n>
struct VecMaskCast<
    int64_t,
    dst_n,
    mask_t,
    mask_n,
    typename std::enable_if_t<
        (dst_n == 2 * mask_n) &&
        (std::is_same_v<mask_t, float> || std::is_same_v<mask_t, int>)>>
{
    static inline VecMask<int64_t, dst_n> apply(
        const VecMask<mask_t, mask_n>& vec_mask)
    {
        VectorizedN<int64_t, dst_n> result;

        for (int i = 0; i < mask_n; ++i) {
            // Load lane as raw bits
            Vectorized<int> int_vec = safe_bitcast<Vectorized<int>>(vec_mask[i]);
            // Wrap into VectorizedN<int,1> because convert expects VectorizedN
            VectorizedN<int, 1> in_int_n;
            in_int_n[0] = int_vec;
			// Call convert to get two Vectorized<int64_t> preserving lane mapping
            auto int64_vecs = convert<int64_t, 2, int, 1>(in_int_n);

            // Store results into result[2*i], result[2*i+1]
		    result[2 * i]     = int64_vecs[0];
            result[2 * i + 1] = int64_vecs[1];
        }
        return VecMask<int64_t, dst_n>(result);
    }
};


#endif

} // namespace CPU_CAPABILITY
} // namespace at::vec
