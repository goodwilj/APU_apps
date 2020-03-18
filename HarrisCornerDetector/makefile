all: HarrisCornerParallel.o HarrisCornerSerial.o

HarrisCornerParallel.o: HarrisCornerParallel.c image.c
	mpic++ HarrisCornerParallel.c image.c -o HarrisCornerParallel.o


HarrisCornerSerial.o: HarrisCornerSerial.c image.c
	g++ HarrisCornerSerial.c image.c -o HarrisCornerSerial.o

runp:
	mpirun -np 5 -q ./HarrisCornerParallel.o

runs:
	./HarrisCornerSerial.o

clean:
	-rm -f HarrisCorner*.o