all: SVMParallel.o SVMSerial.o

SVMParallel.o: SVMParallel.cpp
	mpic++ SVMParallel.cpp -o SVMParallel.o

SVMSerial.o: SVMSerial.cpp
	g++ SVMSerial.cpp -o SVMSerial.o

runp:
	mpirun -np 5 -q ./SVMParallel.o

runs:
	./SVMSerial.o

clean:
	-rm -f SVM*.o