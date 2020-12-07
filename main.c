#include <immintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  uint16_t bf_type;       /* 2 bytes */
  uint32_t bf_size;       /* 4 bytes */
  uint16_t bf_reserved1;  /* 2 bytes */
  uint16_t bf_reserved2;  /* 2 bytes */
  uint32_t bf_offbits;    /* 4 bytes */
} __attribute__((__packed__))
BitMapFileHeader;

typedef struct {
  uint32_t bi_size;             /* 4 bytes */
  int32_t bi_width;             /* 4 bytes */
  int32_t bi_height;            /* 4 bytes */
  uint16_t bi_planes;           /* 2 bytes */
  uint16_t bi_bit_count;        /* 2 bytes */
  uint32_t bi_compression;      /* 4 bytes */
  uint32_t bi_size_image;       /* 4 bytes */
  int32_t bi_x_pix_per_meter;   /* 4 bytes */
  int32_t bi_y_pix_per_meter;   /* 4 bytes */
  uint32_t bi_clr_used;         /* 4 bytes */
  uint32_t bi_clr_important;    /* 4 bytes */
} __attribute__((__packed__))
BitMapInfoHeader;

__m128i divide_si16_by_255(__m128i x) {
  return _mm_srli_epi16(_mm_adds_epu16(x, _mm_srli_epi16(
      _mm_adds_epu16(x, _mm_set1_epi16(0x0101)), 8)), 8);
}
__m128i create_new_pixel(__m128i pixels1,
                         __m128i pixels2,
                         __m128i alpha) {
  __m128i max = _mm_set1_epi16(255);

  __m128i pixels1_hi = _mm_srli_epi16(pixels1, 8);
  __m128i pixels2_hi = _mm_srli_epi16(pixels2, 8);
  __m128i alpha_hi = _mm_srli_epi16(alpha, 8);
  __m128i inv_alpha_hi = _mm_sub_epi16(_mm_set1_epi16(255), alpha_hi);
  __m128i r_hi = _mm_slli_epi16(_mm_min_epi16(divide_si16_by_255(
      _mm_add_epi16(_mm_mullo_epi16(pixels1_hi, inv_alpha_hi),
                    _mm_mullo_epi16(pixels2_hi, alpha_hi))), max), 8);

  __m128i pixels1_lo = _mm_srli_epi16(_mm_slli_epi16(pixels1, 8), 8);
  __m128i pixels2_lo = _mm_srli_epi16(_mm_slli_epi16(pixels2, 8), 8);
  __m128i alpha_lo = _mm_srli_epi16(_mm_slli_epi16(alpha, 8), 8);
  __m128i inv_alpha_lo = _mm_sub_epi16(_mm_set1_epi16(255), alpha_lo);
  __m128i r_lo = _mm_min_epi16(divide_si16_by_255(_mm_add_epi16(
      _mm_mullo_epi16(pixels1_lo, inv_alpha_lo),
      _mm_mullo_epi16(pixels2_lo, alpha_lo))), max);

  return _mm_or_si128(_mm_or_si128(r_hi, r_lo), _mm_set1_epi32(255));
}

uint32_t
read_info(const char* file_name,
          BitMapFileHeader* bf,
          BitMapInfoHeader* bi,
          uint8_t** add_info,
          uint8_t** image) {
  FILE* source = fopen(file_name, "rb");
  fread(bf, sizeof(BitMapFileHeader), 1, source);
  fread(bi, sizeof(BitMapInfoHeader), 1, source);
  uint32_t add_info_size = bf->bf_offbits - sizeof(BitMapInfoHeader) -
                           sizeof(BitMapFileHeader);
  *add_info = calloc(add_info_size, sizeof(uint8_t));
  fread(*add_info, add_info_size, 1, source);
  *image = calloc(bi->bi_size_image, sizeof(uint8_t));
  fread(*image, bi->bi_size_image, 1, source);
  fclose(source);
  return add_info_size;
}

int
main(int argc, char** argv) {
  BitMapFileHeader bf_source_1;
  BitMapInfoHeader bi_source_1;
  uint8_t* image1;
  uint8_t* add_info_1;
  uint32_t add_info_1_size = read_info(argv[1],
                                       &bf_source_1,
                                       &bi_source_1,
                                       &add_info_1,
                                       &image1);
  BitMapFileHeader bf_source_2;
  BitMapInfoHeader bi_source_2;
  uint8_t* image2;
  uint8_t* add_info_2;
  uint32_t add_info_2_size = read_info(argv[2],
                                       &bf_source_2,
                                       &bi_source_2,
                                       &add_info_2,
                                       &image2);

  FILE* output = fopen(argv[3], "wb");
  fwrite(&bf_source_1, sizeof(BitMapFileHeader), 1, output);
  fwrite(&bi_source_1, sizeof(BitMapInfoHeader), 1, output);
  fwrite(add_info_1, add_info_1_size, 1, output);

  uint32_t width1 = bi_source_1.bi_width*4;
  uint32_t width2 = bi_source_2.bi_width*4;
  for (uint32_t i = 0; i < bi_source_2.bi_height; ++i) {
    for (uint32_t j = 0; j < width2; j += 16) {
      uint32_t offset1 = width1*i + j;
      uint32_t offset2 = width2*i + j;
      __m128i pixels1 = _mm_lddqu_si128((const __m128i*)(image1 + offset1));
      __m128i pixels2 = _mm_lddqu_si128((const __m128i*)(image2 + offset2));
      uint8_t alpha[4] = {
          _mm_extract_epi8(pixels2, 0),
          _mm_extract_epi8(pixels2, 4),
          _mm_extract_epi8(pixels2, 8),
          _mm_extract_epi8(pixels2, 12)
      };
      __m128i mask = _mm_set_epi8(
          alpha[3], alpha[3], alpha[3], alpha[3],
          alpha[2], alpha[2], alpha[2], alpha[2],
          alpha[1], alpha[1], alpha[1], alpha[1],
          alpha[0], alpha[0], alpha[0], alpha[0]);
      _mm_storeu_si128((__m128i_u*)(image1 + offset1),
                       create_new_pixel(pixels1, pixels2, mask));
    }
  }
  fwrite(image1, bi_source_1.bi_size_image, 1, output);

  free(add_info_1);
  free(add_info_2);
  free(image1);
  free(image2);
  fclose(output);

  return 0;
}
