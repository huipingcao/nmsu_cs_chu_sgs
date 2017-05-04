VERSION=0.2
# CS Machines
CXX = g++ #-7.2
MPICXX = mpicxx #.mpich2

#Maverick
#CXX=icc
#MPICXX=mpicxx

DEBUGCXXFLAGS = -fPIC -O3 -fopenmp -std=c++11
CXXFLAGS = -fPIC -O3 -march=native -DNDEBUG -fopenmp -std=c++11 -g

# TBB related paths
TBBROOT = tbb
TBBLIB = -L${TBBROOT}/lib -ltbb -ltbbmalloc

INCLUDEPATH = -I${TBBROOT}/include
LIBS = ${TBBLIB} -lrt -lm

# lda-converter f+nomad-lda #setCover_lda  setCover 
all:  splda setcover_subsampling  setCover_lda

splda: splda.cpp splda.h sub_lda.h sub_splda.h sub_flda_d.h sub_alias_lda.h 
	${CXX} ${CXXFLAGS} ${INCLUDEPATH} splda.cpp -o splda 

# setCover:splda.h SetCover.cpp
# 	${CXX} ${CXXFLAGS} ${INCLUDEPATH} SetCover.cpp -o setCover

setCover_lda:splda.h SetCover.h setCover_lda.cpp 
	${CXX} ${CXXFLAGS} ${INCLUDEPATH} setCover_lda.cpp -o setCover_lda

setcover_subsampling: setcover_subsampling.cpp splda.h SetCover.h 
	${CXX} ${CXXFLAGS} ${INCLUDEPATH} setcover_subsampling.cpp -o setcover_subsampling

f+nomad-lda: dist-lda-heap.h dist-lda-heap.cpp sparse_matrix.h petsc-reader.h tbb/lib
	${MPICXX} ${CXXFLAGS} ${INCLUDEPATH} -o f+nomad-lda dist-lda-heap.cpp ${LIBS}

lda-converter: converter.cpp sparse_matrix.h
	${CXX} ${CXXFLAGS} -o lda-converter converter.cpp

tbb/lib:
	make -C tbb/ 

tar: moreclean
	cd ..; mv ${VERSION} nomad-lda-exp-${VERSION}; tar cvzf nomad-lda-exp-${VERSION}.tgz nomad-lda-exp-${VERSION}; mv nomad-lda-exp-${VERSION} ${VERSION}

clean:
	rm -rf lda-converter splda setcover_subsampling f+nomad-lda setCover setCover_lda
	
moreclean: clean
	make -C tbb moreclean
	make -C data moreclean
