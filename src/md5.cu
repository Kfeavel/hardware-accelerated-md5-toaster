#include "main.h"
#include <chrono>
#include <cuda.h>
#include <stdio.h>

#define UINT4 uint
#define MD5_INPUT_LENGTH 512

extern __shared__ unsigned int
    words[]; // shared memory where hash will be stored
__constant__ unsigned int
    target_hash[4]; // constant has we will be searching for

/* F, G and H are basic MD5 functions: selection, majority, parity */

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/* ROTATE_LEFT rotates x left n bits */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4 */
/* Rotation is separate from addition to prevent recomputation */
#define FF(a, b, c, d, x, s, ac)                                               \
  {                                                                            \
    (a) += F((b), (c), (d)) + (x) + (UINT4)(ac);                               \
    (a) = ROTATE_LEFT((a), (s));                                               \
    (a) += (b);                                                                \
  }

#define GG(a, b, c, d, x, s, ac)                                               \
  {                                                                            \
    (a) += G((b), (c), (d)) + (x) + (UINT4)(ac);                               \
    (a) = ROTATE_LEFT((a), (s));                                               \
    (a) += (b);                                                                \
  }

#define HH(a, b, c, d, x, s, ac)                                               \
  {                                                                            \
    (a) += H((b), (c), (d)) + (x) + (UINT4)(ac);                               \
    (a) = ROTATE_LEFT((a), (s));                                               \
    (a) += (b);                                                                \
  }

#define II(a, b, c, d, x, s, ac)                                               \
  {                                                                            \
    (a) += I((b), (c), (d)) + (x) + (UINT4)(ac);                               \
    (a) = ROTATE_LEFT((a), (s));                                               \
    (a) += (b);                                                                \
  }

__device__ int strcmp(const char *s1, const char *s2) {
  for (; *s1 == *s2; ++s1, ++s2)
    if (*s1 == 0)
      return 0;
  return *(unsigned char *)s1 < *(unsigned char *)s2 ? -1 : 1;
}

__device__ size_t strlen(const char *str) {
  const char *s;
  for (s = str; *s; ++s) {
  }
  return (s - str);
}

__device__ char *strcpy(char *dest, const char *src) {
  char *save = dest;
  while (*dest++ = *src++)
    ;
  return save;
}

__device__ char *strncpy(char *dest, const char *src, size_t n) {
  char *ret = dest;
  do {
    if (!n--)
      return ret;
  } while (*dest++ = *src++);
  while (n--)
    *dest++ = 0;
  return ret;
}

__device__ void toupper(char *s) {
  for (; *s; s++)
    if (('a' <= *s) && (*s <= 'z'))
      *s = 'A' + (*s - 'a');
}

__device__ char *md5_pad(const char *input) {
  static char md5_padded[MD5_INPUT_LENGTH];
  int x;
  unsigned int orig_input_length;

  if (input == NULL) {
    return NULL;
  }

  // we store the length of the input (in bits) for later

  orig_input_length = strlen(input) * 8;

  // we would like to split the MD5 into 512 bit chunks with a special ending
  // the maximum input we support is currently 512 bits as we are not expecting
  // a string password to be larger than this

  memset(md5_padded, 0, MD5_INPUT_LENGTH);

  for (x = 0; x < strlen(input) && x < 56; x++) {
    md5_padded[x] = input[x];
  }

  md5_padded[x] = 0x80;

  // now we need to append the length in bits of the original message

  *((unsigned long *)md5_padded + 14) = orig_input_length;

  return md5_padded;
}

char *md5_unpad(char *input) {
  static char md5_unpadded[MD5_INPUT_LENGTH];
  unsigned int orig_length;
  int x;

  if (input == NULL) {
    return NULL;
  }

  memset(md5_unpadded, 0, sizeof(md5_unpadded));

  orig_length = (*((unsigned int *)input + 14) / 8);

  strncpy(md5_unpadded, input, orig_length);

  return md5_unpadded;
}

__device__ void md5(uint *in, uint *hash) {
  uint a, b, c, d;

  const uint a0 = 0x67452301;
  const uint b0 = 0xEFCDAB89;
  const uint c0 = 0x98BADCFE;
  const uint d0 = 0x10325476;

  a = a0;
  b = b0;
  c = c0;
  d = d0;

  /* Round 1 */
#define S11 7
#define S12 12
#define S13 17
#define S14 22
  FF(a, b, c, d, in[0], S11, 3614090360);  /* 1 */
  FF(d, a, b, c, in[1], S12, 3905402710);  /* 2 */
  FF(c, d, a, b, in[2], S13, 606105819);   /* 3 */
  FF(b, c, d, a, in[3], S14, 3250441966);  /* 4 */
  FF(a, b, c, d, in[4], S11, 4118548399);  /* 5 */
  FF(d, a, b, c, in[5], S12, 1200080426);  /* 6 */
  FF(c, d, a, b, in[6], S13, 2821735955);  /* 7 */
  FF(b, c, d, a, in[7], S14, 4249261313);  /* 8 */
  FF(a, b, c, d, in[8], S11, 1770035416);  /* 9 */
  FF(d, a, b, c, in[9], S12, 2336552879);  /* 10 */
  FF(c, d, a, b, in[10], S13, 4294925233); /* 11 */
  FF(b, c, d, a, in[11], S14, 2304563134); /* 12 */
  FF(a, b, c, d, in[12], S11, 1804603682); /* 13 */
  FF(d, a, b, c, in[13], S12, 4254626195); /* 14 */
  FF(c, d, a, b, in[14], S13, 2792965006); /* 15 */
  FF(b, c, d, a, in[15], S14, 1236535329); /* 16 */

  /* Round 2 */
#define S21 5
#define S22 9
#define S23 14
#define S24 20
  GG(a, b, c, d, in[1], S21, 4129170786);  /* 17 */
  GG(d, a, b, c, in[6], S22, 3225465664);  /* 18 */
  GG(c, d, a, b, in[11], S23, 643717713);  /* 19 */
  GG(b, c, d, a, in[0], S24, 3921069994);  /* 20 */
  GG(a, b, c, d, in[5], S21, 3593408605);  /* 21 */
  GG(d, a, b, c, in[10], S22, 38016083);   /* 22 */
  GG(c, d, a, b, in[15], S23, 3634488961); /* 23 */
  GG(b, c, d, a, in[4], S24, 3889429448);  /* 24 */
  GG(a, b, c, d, in[9], S21, 568446438);   /* 25 */
  GG(d, a, b, c, in[14], S22, 3275163606); /* 26 */
  GG(c, d, a, b, in[3], S23, 4107603335);  /* 27 */
  GG(b, c, d, a, in[8], S24, 1163531501);  /* 28 */
  GG(a, b, c, d, in[13], S21, 2850285829); /* 29 */
  GG(d, a, b, c, in[2], S22, 4243563512);  /* 30 */
  GG(c, d, a, b, in[7], S23, 1735328473);  /* 31 */
  GG(b, c, d, a, in[12], S24, 2368359562); /* 32 */

  /* Round 3 */
#define S31 4
#define S32 11
#define S33 16
#define S34 23
  HH(a, b, c, d, in[5], S31, 4294588738);  /* 33 */
  HH(d, a, b, c, in[8], S32, 2272392833);  /* 34 */
  HH(c, d, a, b, in[11], S33, 1839030562); /* 35 */
  HH(b, c, d, a, in[14], S34, 4259657740); /* 36 */
  HH(a, b, c, d, in[1], S31, 2763975236);  /* 37 */
  HH(d, a, b, c, in[4], S32, 1272893353);  /* 38 */
  HH(c, d, a, b, in[7], S33, 4139469664);  /* 39 */
  HH(b, c, d, a, in[10], S34, 3200236656); /* 40 */
  HH(a, b, c, d, in[13], S31, 681279174);  /* 41 */
  HH(d, a, b, c, in[0], S32, 3936430074);  /* 42 */
  HH(c, d, a, b, in[3], S33, 3572445317);  /* 43 */
  HH(b, c, d, a, in[6], S34, 76029189);    /* 44 */
  HH(a, b, c, d, in[9], S31, 3654602809);  /* 45 */
  HH(d, a, b, c, in[12], S32, 3873151461); /* 46 */
  HH(c, d, a, b, in[15], S33, 530742520);  /* 47 */
  HH(b, c, d, a, in[2], S34, 3299628645);  /* 48 */

  /* Round 4 */
#define S41 6
#define S42 10
#define S43 15
#define S44 21
  II(a, b, c, d, in[0], S41, 4096336452);  /* 49 */
  II(d, a, b, c, in[7], S42, 1126891415);  /* 50 */
  II(c, d, a, b, in[14], S43, 2878612391); /* 51 */
  II(b, c, d, a, in[5], S44, 4237533241);  /* 52 */
  II(a, b, c, d, in[12], S41, 1700485571); /* 53 */
  II(d, a, b, c, in[3], S42, 2399980690);  /* 54 */
  II(c, d, a, b, in[10], S43, 4293915773); /* 55 */
  II(b, c, d, a, in[1], S44, 2240044497);  /* 56 */
  II(a, b, c, d, in[8], S41, 1873313359);  /* 57 */
  II(d, a, b, c, in[15], S42, 4264355552); /* 58 */
  II(c, d, a, b, in[6], S43, 2734768916);  /* 59 */
  II(b, c, d, a, in[13], S44, 1309151649); /* 60 */
  II(a, b, c, d, in[4], S41, 4149444226);  /* 61 */
  II(d, a, b, c, in[11], S42, 3174756917); /* 62 */
  II(c, d, a, b, in[2], S43, 718787259);   /* 63 */
  II(b, c, d, a, in[9], S44, 3951481745);  /* 64 */

  a += a0;
  b += b0;
  c += c0;
  d += d0;

  hash[0] = a;
  hash[1] = b;
  hash[2] = c;
  hash[3] = d;

  return;
}

__device__ char *constructWord(char *word, uint32_t Caps, char *AppendIndexes) {
  char AppendChar[] = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS"
                      "TUVWXYZ !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
  AppendChar[0] = '\0';

  int n = strlen(word);
  char newWord[24];
  strcpy(newWord, word);
  // Number of permutations is 2^n
  for (int j = 0; j < n; j++) {
    if (((Caps >> j) & 1) == 1) {
      toupper(&(newWord[j]));
    }
  }
  newWord[n] = AppendChar[AppendIndexes[0]];
  if (AppendChar[AppendIndexes[0]] != '\0') {
    newWord[n + 1] = AppendChar[AppendIndexes[1]];
    if (AppendChar[AppendIndexes[1]] != '\0') {
      newWord[n + 2] = AppendChar[AppendIndexes[2]];
      if (AppendChar[AppendIndexes[2]] != '\0') {
        newWord[n + 3] = AppendChar[AppendIndexes[3]];
        if (AppendChar[AppendIndexes[3]] != '\0') {
          newWord[n + 4] = AppendChar[AppendIndexes[4]];
          if (AppendChar[AppendIndexes[4]] != '\0') {
            newWord[n + 5] = AppendChar[AppendIndexes[5]];
            if (AppendChar[AppendIndexes[5]] != '\0') {
              newWord[n + 6] = '0';
            }
          }
        }
      }
    }
  }

  return newWord;
}

__global__ void md5_cuda_calculate(void *wordlist, int wordlist_len,
                                   uint32_t Caps, char *AppendIndexes,
                                   struct device_stats *stats,
                                   int * target_hash, int targets) {
  unsigned int id;
  int * targetHashes[4] = reinterpret_cast<int (*)[4]>(target_hash);
  uint hash[4];
  int x;

  id = (blockIdx.x * blockDim.x) +
       threadIdx.x; // get our thread unique ID in this run
  unsigned int stride = blockDim.x * gridDim.x;

  for (int i = id; i < wordlist_len; i += stride) {
    char *word = (reinterpret_cast<char(*)[64]>(wordlist))[i];
    char *changed = constructWord(word, Caps, AppendIndexes);
    char *padWord = md5_pad(changed);
    md5(static_cast<uint *>(padWord), hash);

    for (int i = 0; i < targets; i++) {
      if (hash[0] == targetHashes[i][0] && hash[1] == targetHashes[i][1] &&
          hash[2] == targetHashes[i][2] && hash[3] == targetHashes[i][3]) {

        // !! WE HAVE A MATCH !!
        stats.hash[stats.hash_found] = i;
        for (x = 0; x < 64; x++) {
          // copy the matched word across
          stats->word[stats.hash_found][x] = *(char *)((char *)padWord + x);
        }
        stats->hash_found++;
      }
    }
  }
}

void md5_calculate(struct cuda_device *device) {
  cudaEvent_t start, stop;
  float time;
  uint128_t hashes = 0;
  uint32_t Caps = 0;
  int foundHashes = 0;
  auto start = chrono::high_resolution_clock::now();
  char *AppendIndexes;
  cudaMallocManaged(&AppendIndexes, 6 * sizeof(char));
  memset(AppendIndexes, 0, 6 * sizeof(char)); // start everything as 0
  for (int sixth = 0; sixth < 96; sixth++) {
    for (int fifth = 0; fifth < 96; fifth++) {
      for (int fourth = 0; fourth < 96; fourth++) {
        for (int third = 0; third < 96; third++) {
          for (int second = 0; second < 96; second++) {
            for (int first = 0; first < 96; first++) {
              for (Caps = 0; Caps < (1 << 16); Caps++) {
                AppendIndexes[0] = first;
                AppendIndexes[1] = second;
                AppendIndexes[2] = third;
                AppendIndexes[3] = fourth;
                AppendIndexes[4] = fifth;
                AppendIndexes[5] = sixth;

                md5_cuda_calculate<<<device->max_blocks, device->max_threads>>>(
                    device.wordlist, device.wordlist_len, Caps, AppendIndexes,
                    static_cast<device_stats *>(device.device_stats_memory),
                    reinterpret_cast<int(*)[4]>(device.target_hash),
                    device.targets);

                cudaDeviceSynchronize();
                hashes += device.wordlist_len;
                if (foundHashes < device.stats.hash_found) {
                  auto now = chrono::high_resolution_clock::now();
                  using day_t = duration<long, std::ratio<3600 * 24>>;
                  auto dur = end - start;
                  auto d = duration_cast<day_t>(dur);
                  auto h = duration_cast<hours>(dur -= d);
                  auto m = duration_cast<minutes>(dur -= h);
                  auto s = duration_cast<seconds>(dur -= m);
                  auto ms = duration_cast<milliseconds>(dur -= s);
                  for (int i = (device.stats.hash_found - foundHashes); i > 0;
                       i--) {
                    char *unpadHash = md5_unpad(
                        device.stats.word[device.stats.hash_found - i]);
                    printf("%i:\t%c:\t", device.stats.hash[i], unpadHash);
                  }
                }
                printf("")
              }
            }
          }
        }
      }
    }
  }
}
