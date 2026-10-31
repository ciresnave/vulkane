// Throwaway repro for vulkane#130: N threads, each creates its own
// instance + device and resolves every command name through
// vkGetDeviceProcAddr, then tears down. No Rust involved.
// usage: loader_repro <names.txt> <threads> <iters> <mode>
//   mode 0: no locking   mode 1: one global mutex around the whole body
//   mode 2: global mutex around only vkGetDeviceProcAddr resolution
#include <vulkan/vulkan.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char **names;
static int nnames;
static int iters, mode;
static pthread_mutex_t big = PTHREAD_MUTEX_INITIALIZER;

static void body(int do_lock_resolve) {
  VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                          .apiVersion = VK_API_VERSION_1_1};
  VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .pApplicationInfo = &ai};
  VkInstance inst;
  if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { fprintf(stderr, "no instance\n"); exit(3); }
  uint32_t n = 1;
  VkPhysicalDevice pd;
  vkEnumeratePhysicalDevices(inst, &n, &pd);
  float prio = 1.0f;
  VkDeviceQueueCreateInfo q = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                               .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio};
  VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                            .queueCreateInfoCount = 1, .pQueueCreateInfos = &q};
  VkDevice dev;
  if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { fprintf(stderr, "no device\n"); exit(3); }
  PFN_vkGetDeviceProcAddr gdpa =
      (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(inst, "vkGetDeviceProcAddr");
  if (do_lock_resolve) pthread_mutex_lock(&big);
  for (int i = 0; i < nnames; i++) (void)gdpa(dev, names[i]);
  if (do_lock_resolve) pthread_mutex_unlock(&big);
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);
}

static void *worker(void *arg) {
  (void)arg;
  for (int i = 0; i < iters; i++) {
    if (mode == 1) { pthread_mutex_lock(&big); body(0); pthread_mutex_unlock(&big); }
    else body(mode == 2);
  }
  return NULL;
}

int main(int argc, char **argv) {
  if (argc < 5) return 2;
  FILE *f = fopen(argv[1], "r");
  char line[256];
  names = malloc(sizeof(char *) * 4096);
  while (fgets(line, sizeof line, f)) {
    line[strcspn(line, "\r\n")] = 0;
    if (*line) names[nnames++] = strdup(line);
  }
  int threads = atoi(argv[2]);
  iters = atoi(argv[3]);
  mode = atoi(argv[4]);
  pthread_t t[64];
  for (int i = 0; i < threads; i++) pthread_create(&t[i], NULL, worker, NULL);
  for (int i = 0; i < threads; i++) pthread_join(t[i], NULL);
  return 0;
}
