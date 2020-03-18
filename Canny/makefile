all: CannyEdgeSerial.o CannyEdgeParallel.o

CannyEdgeSerial.o: CannyEdgeSerialNoStruct.c image.c
	gcc CannyEdgeSerialNoStruct.c image.c -fopenmp -o CannyEdgeSerial.o -lm


CannyEdgeParallel.o: CannyEdgeMPIwNMS.c image.c
	mpicc CannyEdgeMPIwNMS.c image.c -fopenmp -o CannyEdgeParallel.o -lm

runp:
	mpirun -np 5 -q ./CannyEdgeParallel.o

runs:
	./CannyEdgeSerial.o

clean:
	-rm -f CannyEdge*.o