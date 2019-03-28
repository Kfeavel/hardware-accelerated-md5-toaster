#include "main.h"
#include <algorithm>
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <string>

#define MD5_INPUT_LENGTH 512

extern void md5_calculate(struct cuda_device *);

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

char *md5_pad(char *input) {
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

int get_cuda_device(struct cuda_device *device) {
  int device_count;

  if (cudaGetDeviceCount(&device_count) != CUDA_SUCCESS) {
    // cuda not supported
    return -1;
  }

  while (device_count >= 0) {
    if (cudaGetDeviceProperties(&device->prop, device_count) == CUDA_SUCCESS) {
      // we have found our device
      device->device_id = device_count;
      return device_count;
    }

    device_count--;
  }

  return -1;
}

#define REQUIRED_SHARED_MEMORY 64
#define FUNCTION_PARAM_ALLOC 256

int calculate_cuda_params(struct cuda_device *device) {
  int max_threads;
  int max_blocks;
  int shared_memory;

  max_threads = device->prop.maxThreadsPerBlock;
  shared_memory = device->prop.sharedMemPerBlock - FUNCTION_PARAM_ALLOC;

  // calculate the most threads that we can support optimally

  while ((shared_memory / max_threads) < REQUIRED_SHARED_MEMORY) {
    max_threads--;
  }

  // now we spread our threads across blocks

  max_blocks = 40; // ?? we need to calculate this !

  device->max_threads = max_threads;     // most threads we support
  device->max_blocks = max_blocks;       // most blocks we support
  device->shared_memory = shared_memory; // shared memory required

  // now we need to have (device.max_threads * device.max_blocks) number of
  // words in memory for the graphics card

  device->device_global_memory_len =
      (device->max_threads * device->max_blocks) * 64;

  return 1;
}

struct wordlist_file {
  ifstream ifs;
  int len;
  char **words;
};

#define WORDS_TO_CACHE 10000
#define FILE_BUFFER 512

#define CRLF 2
#define LF 1

int read_wordlist(struct wordlist_file *file) {
  map<string, int> mp;

  std::string word;
  while (file->ifs >> word) {
    transform(word.begin(), word.end(), word.begin(), ::tolower);
    if (!mp.count(word)) {
      mp.insert(mape_pair(word, 1));
    } else {
      mp[word]++;
    }
  }
  file->len = 0;
  for (auto p = mp.begin(); p != mp.end(); p++) {
    strcpy(file->words[p], md5_pad(word.c_str()));
    len++;
  }

  return 1;
}

int read_hashlist(ifstream ifs) {}

/*********************************************************************
 *  TAKEN FROM: http://www.codeproject.com/KB/string/hexstrtoint.aspx
 *
 *  has been slightly modified
 *
 *  Many Thanks Anders Molin
 *********************************************************************/

struct CHexMap {
  char chr;
  int value;
};

#define true 1
#define false 0

#define HexMapL 22

int _httoi(const char *value) {
  struct CHexMap HexMap[HexMapL] = {
      {'0', 0},  {'1', 1},  {'2', 2},  {'3', 3},  {'4', 4},  {'5', 5},
      {'6', 6},  {'7', 7},  {'8', 8},  {'9', 9},  {'A', 10}, {'B', 11},
      {'C', 12}, {'D', 13}, {'E', 14}, {'F', 15}, {'a', 10}, {'b', 11},
      {'c', 12}, {'d', 13}, {'e', 14}, {'f', 15},
  };
  int i;

  char *mstr = strdup(value);
  char *s = mstr;
  int result = 0;
  int found = false;

  if (*s == '0' && *(s + 1) == 'X') {
    s += 2;
  }

  int firsttime = true;

  while (*s != '\0') {
    for (i = 0; i < HexMapL; i++) {

      if (*s == HexMap[i].chr) {

        if (!firsttime) {
          result <<= 4;
        }

        result |= HexMap[i].value;
        found = true;
        break;
      }
    }

    if (!found) {
      break;
    }

    s++;
    firsttime = false;
  }

  free(mstr);
  return result;
}

/*************************************************************************/

void print_info(void) {
  printf("cuda_md5_crack programmed by XPN "
         "(http://xpnsbraindump.blogspot.com)\n\n");
  return;
}

#define ARG_MD5 2
#define ARG_WORDLIST 1
#define ARG_COUNT 1 + 2

int main(int argc, char **argv) {
  char *output;
  int x;
  int y;
  struct cuda_device device;
  int available_words = 1;
  int current_words = 0;
  struct wordlist_file file;
  char input_hash[4][9];

  print_info();

  if (argc != ARG_COUNT) {
    printf("Usage: %s WORDLIST_FILE MD5_HASH\n", argv[0]);
    return -1;
  }
  file.ifs.open(argv[ARG_WORDLIST]);
  if (file.ifs.is_open()) {
    printf("Error Opening Wordlist File: %s\n", argv[ARG_WORDLIST]);
    return -1;
  }

  if (read_wordlist(&file) == 0) {
    printf("Words in the wordlist file: %s\n", argv[ARG_WORDLIST]);
    return -1;
  } else {
    printf("%s contains %i words", argv[ARG_WORDLIST], file.len);
  }

  // first things first, we need to select our CUDA device

  if (get_cuda_device(&device) == -1) {
    printf("No Cuda Device Installed\n");
    return -1;
  }

  // we now need to calculate the optimal amount of threads to use for this card

  calculate_cuda_params(&device);

  // now we input our target hash

  if (argv[ARG_MD5]) {
    printf("Not a valid MD5 Hash (should be 32 bytes and only Hex Chars\n");
    return -1;
  }

  // we split the input hash into 4 blocks

  memset(input_hash, 0, sizeof(input_hash));

  for (x = 0; x < 4; x++) {
    strncpy(input_hash[x], argv[ARG_MD5] + (x * 8), 8);
    device.target_hash[x] = htonl(_httoi(input_hash[x]));
  }

  // allocate global memory for use on device
  if (cudaMalloc(&device.device_global_memory,
                 device.device_global_memory_len) != CUDA_SUCCESS) {
    printf("Error allocating memory on device (global memory)\n");
    return -1;
  }

  // allocate the 'stats' that will indicate if we are successful in cracking
  if (cudaMalloc(&device.device_stats_memory, sizeof(struct device_stats)) !=
      CUDA_SUCCESS) {
    printf("Error allocating memory on device (stats memory)\n");
    return -1;
  }

  // allocate debug memory if required
  if (cudaMalloc(&device.device_debug_memory,
                 device.device_global_memory_len) != CUDA_SUCCESS) {
    printf("Error allocating memory on device (debug memory)\n");
    return -1;
  }

  // make sure the stats are clear on the device
  if (cudaMemset(device.device_stats_memory, 0, sizeof(struct device_stats)) !=
      CUDA_SUCCESS) {
    printf("Error Clearing Stats on device\n");
    return -1;
  }

  // this is our host memory that we will copy to the graphics card
  if ((device.host_memory = malloc(device.device_global_memory_len)) == NULL) {
    printf("Error allocating memory on host\n");
    return -1;
  }

  // put our target hash into the GPU constant memory as this will not change
  // (and we can't spare shared memory for speed)
  if (cudaMemcpyToSymbol("target_hash", device.target_hash, 16*15, 0,
                         cudaMemcpyHostToDevice) != CUDA_SUCCESS) {
    printf("Error initalizing constants\n");
    return -1;
  }

  int z;

  while (available_words) {
    memset(device.host_memory, 0, device.device_global_memory_len);

    for (x = 0; x < (device.device_global_memory_len / 64) &&
                file.words[current_words] != (char *)0;
         x++, current_words++)
	{
      output = md5_pad(file.words[current_words]);
      memcpy(device.host_memory + (x * 64), output, 64);
    }

    if (file.words[current_words] == (char *)0) {
      // read some more words !
      current_words = 0;
      if (!read_wordlist(&file)) {
        // no more words available
        available_words = 0;
        // we continue as we want to flush the cache !
      }
    }

    // now we need to transfer the MD5 hashes to the graphics card for
    // preperation

    if (cudaMemcpy(device.device_global_memory, device.host_memory,
                   device.device_global_memory_len,
                   cudaMemcpyHostToDevice) != CUDA_SUCCESS) {
      printf("Error Copying Words to GPU\n");
      return -1;
    }

    md5_calculate(&device); // launch the kernel of the CUDA device

    if (cudaMemcpy(&device.stats, device.device_stats_memory,
                   sizeof(struct device_stats),
                   cudaMemcpyDeviceToHost) != CUDA_SUCCESS) {
      printf("Error Copying STATS from the GPU\n");
      return -1;
    }

    if (device.stats.hash_found == 1) {
      printf("WORD FOUND: [%s]\n",
             md5_unpad(
                 static_cast<char *>(static_cast<void *>(device.stats.word))));
      break;
    }
  }

  if (device.stats.hash_found != 1) {
    printf("No word could be found for the provided MD5 hash\n");
  }


}
