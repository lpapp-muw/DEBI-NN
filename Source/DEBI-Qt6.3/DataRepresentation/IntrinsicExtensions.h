/*!
 * \file
 * Extension for intrinsic functions. This file is part of Datarepresentation module.
 *
 * \remarks
 *
 * \authors
 * lpapp
 */

#pragma once

#include <DataRepresentation/IntrinsicFunctions.h>

//-----------------------------------------------------------------------------

// Multiplies floating-point values by 2.0 using 128 bit vector: 2 * A.
static inline __m128 _mmx2_ps( __m128 __A );
static inline __m128d _mmx2_pd( __m128d __A );

// Squares floating-point values using 128 bit vector: A * A.
static inline __m128 _mmsqr_ps( __m128 __A );
static inline __m128d _mmsqr_pd( __m128d __A );

// Calculates absolute value of single-precision floating-point values using 128 bit vector: | A |.
static inline __m128 _mmabs_ps( __m128 __A );
static inline __m128d _mmabs_pd( __m128d __A );

// Selects the bits of either A or B depending of the bits of C. (Bitwise blend.)
static inline __m128 _mmblendb_ps( __m128 __A, __m128 __B, __m128 __C );
static inline __m128d _mmblendb_ps( __m128d __A, __m128d __B, __m128d __C );

#if defined( adr_USE_AVX )
// Multiplies floating-point values by 2.0 using 256 bit vector: 2 * A.
static inline __m256 _mm256_x2_ps( __m256 __A );
static inline __m256d _mm256_x2_pd( __m256d __A );

// Squares floating-point values using 256 bit vector: A * A.
static inline __m256 _mm256_sqr_ps( __m256 __A );
static inline __m256d _mm256_sqr_pd( __m256d __A );

// Calculates absolute value of floating-point values using 256 bit vector: | A |.
static inline __m256 _mm256_abs_ps( __m256 __A );
static inline __m256d _mm256_abs_pd( __m256d __A );

// Selects the value of either A or B depending of the bits of C. (Bitwise blend.)
static inline __m256 _mm256_blendb_ps( __m256 __A, __m256 __B, __m256 __C );
static inline __m256d _mm256_blendb_ps( __m256d __A, __m256d __B, __m256d __C );
#endif

//-----------------------------------------------------------------------------

/*!
 * \brief Multiplies single-precision floating-point values by 2.0 using 128 bit float vector.
 *
 * \param [in] __A Float vector.
 * \return 2 * A.
 */
static inline __m128 _mmx2_ps( __m128 __A )
{
	return _mm_add_ps( __A, __A );
}

/*!
 * \brief Multiplies double-precision floating-point values by 2.0 using 128 bit double vector.
 *
 * \param [in] __A Double vector.
 * \return 2 * A.
 */
static inline __m128d _mmx2_pd( __m128d __A )
{
	return _mm_add_pd( __A, __A );
}

/*!
 * \brief Squares single-precision floating-point values using 128 bit float vector.
 *
 * \param [in] __A Float vector.
 * \return A * A.
 */
static inline __m128 _mmsqr_ps( __m128 __A )
{
	return _mm_mul_ps( __A, __A );
}

/*!
 * \brief Squares double-precision floating-point values using 128 bit double vector.
 *
 * \param [in] __A Double vector.
 * \return A * A.
 */
static inline __m128d _mmsqr_pd( __m128d __A )
{
	return _mm_mul_pd( __A, __A );
}

/*!
 * \brief Calculates absolute value of single-precision floating-point values using 128 bit float vector.
 *
 * \param [in] __A Float vector.
 * \return | A |.
 */
static inline __m128 _mmabs_ps( __m128 __A )
{
	return _mm_and_ps( __A, _mm_castsi128_ps( _mm_set1_epi32( 0x7FFFFFFF ) ) );
}

/*!
 * \brief Calculates absolute value of double-precision floating-point values using 128 bit double vector.
 *
 * \param [in] __A Double vector.
 * \return | A |.
 */
static inline __m128d _mmabs_pd( __m128d __A )
{
	return _mm_and_pd( __A, _mm_castsi128_pd( _mm_set1_epi32( 0x7FFFFFFF ) ) );
}

/*!
 * \brief Selects the bits of either A or B depending of the bits of C. (Bitwise blend.)
 *
 * \param [in] __A Float vector.
 * \param [in] __B Float vector.
 * \param [in] __C Bitmask vector.
 * \return A & ~C | B & C.
 */
static inline __m128 _mmblendb_ps( __m128 __A, __m128 __B, __m128 __C )
{
	return _mm_or_ps( _mm_andnot_ps( __C, __A ), _mm_and_ps( __C, __B ) );
}

/*!
 * \brief Selects the bits of either A or B depending of the bits of C. (Bitwise blend.)
 *
 * \param [in] __A Double vector.
 * \param [in] __B Double vector.
 * \param [in] __C Bitmask vector.
 * \return A & ~C | B & C.
 */
static inline __m128d _mmblendb_ps( __m128d __A, __m128d __B, __m128d __C )
{
	return _mm_or_pd( _mm_andnot_pd( __C, __A ), _mm_and_pd( __C, __B ) );
}

#if defined( adr_USE_AVX )
/*!
 * \brief Multiplies single-precision floating-point values by 2.0 using 256 bit float vector.
 *
 * \param [in] __A Float vector.
 * \return 2 * A.
 */
static inline __m256 _mm256_x2_ps( __m256 __A )
{
	return _mm256_add_ps( __A, __A );
}

/*!
 * \brief Multiplies double-precision floating-point values by 2.0 using 256 bit double vector.
 *
 * \param [in] __A Double vector.
 * \return 2 * A.
 */
static inline __m256d _mm256_x2_pd( __m256d __A )
{
	return _mm256_add_pd( __A, __A );
}

/*!
 * \brief Squares single-precision floating-point values using 256 bit float vector.
 *
 * \param [in] __A Float vector.
 * \return A * A.
 */
static inline __m256 _mm256_sqr_ps( __m256 __A )
{
	return _mm256_mul_ps( __A, __A );
}

/*!
 * \brief Squares double-precision floating-point values using 256 bit double vector.
 *
 * \param [in] __A Double vector.
 * \return A * A.
 */
static inline __m256d _mm256_sqr_pd( __m256d __A )
{
	return _mm256_mul_pd( __A, __A );
}

/*!
 * \brief Calculates absolute value of single-precision floating-point values using 256 bit float vector.
 *
 * \param [in] __A Float vector.
 * \return | A |.
 */
static inline __m256 _mm256_abs_ps( __m256 __A )
{
	return _mm256_and_ps( __A, _mm256_castsi256_ps( _mm256_set1_epi32( 0x7FFFFFFF ) ) );
}

/*!
 * \brief Calculates absolute value of double-precision floating-point values using 256 bit double vector.
 *
 * \param [in] __A Float vector.
 * \return | A |.
 */
static inline __m256d _mm256_abs_pd( __m256d __A )
{
	return _mm256_and_pd( __A, _mm256_castsi256_pd( _mm256_set1_epi32( 0x7FFFFFFF ) ) );
}

/*!
 * \brief Selects the bits of either A or B depending of the bits of C. (Bitwise blend.)
 *
 * \param [in] __A Float vector.
 * \param [in] __B Float vector.
 * \param [in] __C Bitmask vector.
 * \return A & ~C | B & C.
 */
static inline __m256 _mm256_blendb_ps( __m256 __A, __m256 __B, __m256 __C )
{
	return _mm256_or_ps( _mm256_andnot_ps( __C, __A ), _mm256_and_ps( __C, __B ) );
}

/*!
 * \brief Selects the bits of either A or B depending of the bits of C. (Bitwise blend.)
 *
 * \param [in] __A Double vector.
 * \param [in] __B Double vector.
 * \param [in] __C Bitmask vector.
 * \return A & ~C | B & C.
 */
static inline __m256d _mm256_blendb_ps( __m256d __A, __m256d __B, __m256d __C )
{
	return _mm256_or_pd( _mm256_andnot_pd( __C, __A ), _mm256_and_pd( __C, __B ) );
}
#endif

//-----------------------------------------------------------------------------
