#!/bash/bin

CXX = g++
CXXFLAGS = -std=c++11 -O2 -lm -lgsl -lgslcblas -fopenmp -lfftw3 #-Wall -Wextra -g
 

prog: main.o lib.o  sdde_solver.o
	$(CXX)  -o $@  $^ $(CXXFLAGS)

main.o : main.cpp lib.h sdde_solver.h
	$(CXX) -c main.cpp  $(CXXFLAGS)

lib.o: lib.cpp lib.h sdde_solver.h 
	$(CXX) -c lib.cpp $(CXXFLAGS)

sdde_solver.o: sdde_solver.cpp sdde_solver.h lib.h
	$(CXX) -c sdde_solver.cpp $(CXXFLAGS)	
	
.PHONEY: clean
clean:
	rm -rf  *.pyc 
	rm -rf ../data/text/* 
	rm -rf ../data/fig/*
	rm -rf ../data/npz/*
	rm -f  *.out *.pyc 
	rm -f ../data/text/networks/*.txt

.PHONEY: eradicate
eradicate: clean
	rm -rf *.o  prog
	rm -rf *.txt
	rm -rf *.out
	rm -rf __pycache__

.PHONY: clean, eradicate
