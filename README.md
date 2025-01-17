# LoneliestSeed-cl


Build command (Mostly stolen from kaktwoos-cl):

g++ -w -m64 -O3 ./main.c -o loneliest-cl.bin -Iboinc/ -Lboinc/lib/lin -static-libgcc -static-libstdc++ -lboinc_api -lboinc -lboinc_opencl -pthread -Wl,-Bdynamic -lOpenCL -Wl,-dynamic-linker,/lib64/ld-linux-x86-64.so.2 -DBOINC