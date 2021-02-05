#if defined(MFEM_USE_UMPIRE) && defined(MFEM_USE_CUDA)

#include "mfem.hpp"
#include "unit_tests.hpp"

#include <unistd.h>
#include <stdio.h>
#include "umpire/Umpire.hpp"
#include <cuda.h>

using namespace mfem;

constexpr unsigned num_elems = 1024;
constexpr unsigned num_bytes = num_elems * sizeof(double);
constexpr double host_val = 1.0;
constexpr double dev_val = 1.0;

static long alloc_size(int id)
{
   auto &rm = umpire::ResourceManager::getInstance();
   auto a                    = rm.getAllocator(id);
   return a.getCurrentSize();
}

static bool is_pinned_host(void * p)
{
   unsigned flags;
   auto err = cudaHostGetFlags(&flags, p);
   if (err == cudaSuccess) { return true; }
   else if (err == cudaErrorInvalidValue) { return false; }
   fprintf(stderr, "fatal (is_pinned_host): unknown return value: %d\n", err);
   return false;
}

static void test_umpire_device_memory()
{
   auto &rm = umpire::ResourceManager::getInstance();

   const int permanent = umpire::Allocator(
                            rm.makeAllocator<
                            umpire::strategy::DynamicPoolMap,
                            true>("MFEM-Permanent-Device-Pool",
                                  rm.getAllocator("DEVICE"), 0, 0))
                         .getId();

   const int temporary = umpire::Allocator(
                            rm.makeAllocator<
                            umpire::strategy::DynamicPoolList,
                            true>("MFEM-Temporary-Device-Pool",
                                  rm.getAllocator("DEVICE"), 0, 0))
                         .getId();

   Device::SetHostUmpire(false);
   Device::SetDeviceUmpire(true);
   MemoryManager::SetUmpireDeviceAllocatorId(permanent);
   MemoryManager::SetUmpireDeviceTempAllocatorId(temporary);
   Device device("cuda");

   printf("Both pools should be empty at startup: ");
   REQUIRE(alloc_size(permanent) == 0);
   REQUIRE(alloc_size(temporary) == 0);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // allocate on host, use permanent device memory when needed
   Vector host_perm(num_elems);
   REQUIRE(!is_pinned_host(host_perm.GetData()));
   // allocate on host, use temporary device memory when needed
   Vector host_temp(num_elems); host_temp = host_val; host_temp.UseTemporary(true);
   REQUIRE(!is_pinned_host(host_temp.GetData()));

   printf("Allocated %u bytes on the host, pools should still be empty: ",
          num_bytes*2);
   REQUIRE((alloc_size(permanent) == 0 && alloc_size(temporary) == 0));
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // uses permanent device memory
   host_perm.Write();

   printf("Write of size %u to perm, temp should still be empty: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes)
   REQUIRE(alloc_size(temporary) == 0);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // uses temporary device memory
   double * d_host_temp = host_temp.ReadWrite();
   //MFEM_FORALL(i, num_elems, { d_host_temp[i] = dev_val; });

   printf("Write of size %u to temp: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes)
   REQUIRE(alloc_size(temporary) == num_bytes);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // allocates in permanent device memory
   Vector dev_perm(num_elems, MemoryClass::DEVICE);

   printf("Allocate %u more bytes in permanent memory: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes*2);
   REQUIRE(alloc_size(temporary) == num_bytes);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // allocates in temporary device memory
   // shorten with using statement
   using mc = mfem::MemoryClass;
   Vector dev_temp(num_elems, mc::DEVICE_TEMP);
   //double * d_dev_temp = dev_temp.Write();
   //MFEM_FORALL(i, num_elems, { d_dev_temp[i] = dev_val; });

   printf("Allocate %u more bytes in temporary memory: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes*2)
   REQUIRE(alloc_size(temporary) == num_bytes*2);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // pinned host memory
   Vector pinned_host_perm(num_elems, mfem::MemoryType::HOST_PINNED);
   REQUIRE(is_pinned_host(pinned_host_perm.GetData()));
   Vector pinned_host_temp(num_elems, mfem::MemoryType::HOST_PINNED);
   pinned_host_temp.UseTemporary(true);
   REQUIRE(is_pinned_host(pinned_host_temp.GetData()));
   printf("Allocate %u pinned bytes in on the host: ", num_bytes*2);
   REQUIRE(alloc_size(permanent) == num_bytes*2)
   REQUIRE(alloc_size(temporary) == num_bytes*2);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   pinned_host_perm.Write();
   printf("Allocate %u more bytes in permanent memory: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes*3)
   REQUIRE(alloc_size(temporary) == num_bytes*2);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   pinned_host_temp.Write();
   printf("Allocate %u more bytes in temporary memory: ", num_bytes);
   REQUIRE(alloc_size(permanent) == num_bytes*3)
   REQUIRE(alloc_size(temporary) == num_bytes*3);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));

   // remove from temporary memory
   // don't copy to host, verify that the value is still the "host" value
   host_temp.DeleteDevice(false);
   REQUIRE(host_temp[0] == host_val);
   // copy to host, verify that the value is the "device" value
   dev_temp.DeleteDevice();
   // TODO TMS
   //REQUIRE(dev_temp[0] == dev_val);
   pinned_host_temp.DeleteDevice();

   printf("Delete all temporary memory: ");
   REQUIRE(alloc_size(permanent) == num_bytes*2);
   REQUIRE(alloc_size(temporary) == 0);
   printf("perm=%ld, temp=%ld\n", alloc_size(permanent), alloc_size(temporary));
}

TEST_CASE("MemoryManager", "[MemoryManager]")
{
   SECTION("Umpire")
   {
      test_umpire_device_memory();
   }
}

#endif // MFEM_USE_UMPIRE && MFEM_USE_CUDA