all: NeuralNetworkParallel.o NeuralNetworkSerial.o

NeuralNetworkParallel.o: NeuralNetworkMPI.c
	mpicc NeuralNetworkMPI.c -fopenmp -o NeuralNetworkParallel.o


NeuralNetworkSerial.o: NeuralNetworkSerial.c
	gcc NeuralNetworkSerial.c -fopenmp -o NeuralNetworkSerial.o

runp:
	mpirun -np 5 -q ./NeuralNetworkParallel.o

runs:
	./NeuralNetworkSerial.o

clean:
	rm -f NeuralNetwork*.o