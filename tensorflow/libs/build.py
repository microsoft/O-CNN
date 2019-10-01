import os
import sys
import tensorflow as tf
import subprocess

# OCTREE_DIR = '/home/penwan/workspace/ps/Octree'
OCTREE_DIR = '/mnt/hd3/penwan/Projects/Octree'
CUDA_DIR   = '/usr/local/cuda-10.1'
if len(sys.argv) > 1: OCTREE_DIR = sys.argv[1]
if len(sys.argv) > 2: CUDA_DIR = sys.argv[2]


lines = []

TF_CFLAGS = " ".join(tf.sysconfig.get_compile_flags())
TF_LFLAGS = " ".join(tf.sysconfig.get_link_flags())
lines.append("TF_CFLAGS := %s" % TF_CFLAGS)
lines.append("TF_LFLAGS := %s" % TF_LFLAGS)
lines.append("OCT_CFLAGS := -I%s/octree" % OCTREE_DIR)
lines.append("OCT_LFLAGS := -L%s/build -loctree_lib -lrply" % OCTREE_DIR)
lines.append("")

lines.append("NVCC_FLAGS1 := -std=c++11 -O2 -c")
lines.append("NVCC_FLAGS2 := $(TF_CFLAGS) $(OCT_CFLAGS) -I %s/include -x cu -Xcompiler -fPIC -D GOOGLE_CUDA=1 -I /usr/local -expt-relaxed-constexpr -DNDEBUG" % CUDA_DIR)
lines.append("CC_FLAGS1 := -std=c++11 -O2")
lines.append("CC_FLAGS2 := $(TF_CFLAGS) $(OCT_CFLAGS) -I %s/include -L %s/lib64 -D GOOGLE_CUDA=1 $(TF_LFLAGS) $(OCT_LFLAGS) -fPIC -lcudart" % (CUDA_DIR, CUDA_DIR))
lines.append("")

lines.append("CC := g++")
lines.append("NVCC := %s/bin/nvcc" % CUDA_DIR)
lines.append("")

cpu_objects = []
cpu_sources = []
gpu_objects = []
gpu_sources = []
all_headers = []
for filename in sorted(os.listdir(".")):
    if filename.endswith(".cu.cc") or filename.endswith(".cu"):
        targetname = filename + ".o"
        gpu_sources.append(filename)
        gpu_objects.append(os.path.join("object", targetname))
    elif filename.endswith(".cc") or filename.endswith(".cpp"):
        targetname = filename + ".o"
        cpu_sources.append(filename)
        cpu_objects.append(os.path.join("object", targetname))
    elif filename.endswith(".h"):
        all_headers.append(filename)

lines.append("cpu_objects := %s" % " ".join(cpu_objects))
lines.append("gpu_objects := %s" % " ".join(gpu_objects))
lines.append("")

lines.append(".PHONY : all clean")
lines.append("")

lines.append("all : libocnn.so")
lines.append("")

lines.append("libocnn.so : $(cpu_objects) $(gpu_objects)")
lines.append("\t$(CC) $(CC_FLAGS1) -shared -o libocnn.so object/*.o $(CC_FLAGS2)")
lines.append("")

for i in range(len(cpu_objects)):
    dependency = b" ".join(subprocess.Popen("g++ -std=c++11 -MM -MG %s" % cpu_sources[i], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True).stdout.readlines()).decode("utf-8")
    headers = []
    for h in all_headers:
        if dependency.find(h) > 0:
            headers.append(h)
    headers = " ".join(headers)
    lines.append("%s : %s %s" % (cpu_objects[i], headers, cpu_sources[i]))
    lines.append("\t$(CC) $(CC_FLAGS1) -c %s $(CC_FLAGS2) -o %s" % (cpu_sources[i], cpu_objects[i]))
    lines.append("")

for i in range(len(gpu_objects)):
    dependency = b" ".join(subprocess.Popen("g++ -std=c++11 -MM -MG %s" % gpu_sources[i], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True).stdout.readlines()).decode("utf-8")
    headers = []
    for h in all_headers:
        if dependency.find(h) > 0:
            headers.append(h)
    headers = " ".join(headers)
    lines.append("%s : %s %s" % (gpu_objects[i], headers, gpu_sources[i]))
    lines.append("\t$(NVCC) $(NVCC_FLAGS1) %s $(NVCC_FLAGS2) -o %s" % (gpu_sources[i], gpu_objects[i]))
    lines.append("")

lines.append("clean :")
lines.append("\t rm %s" % os.path.join("object", "*.o"))
lines.append("")

lines = [line + "\n" for line in lines]

with open("Makefile", "w") as f:
    f.writelines(lines)

if not os.path.exists("object"):
    os.mkdir("object")
os.system("make -j all")
