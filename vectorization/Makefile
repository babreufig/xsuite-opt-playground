# Compiler and flags
CC = gcc
CFLAGS = -Wall -O3 -march=native -ftree-vectorize
LDFLAGS = -lm

# Source and output files
SRC = vec_min_track_example.c
OBJ = vec_min_track_example.o
EXEC = vec_min_track_example

# Default target (just compile with -O3 and vectorization analysis)
all: opt-vec opt-vec-missed

# Compile with vectorization analysis
opt-vec: CFLAGS += -fopt-info-vec
opt-vec: $(SRC)
	$(CC) $(CFLAGS) -o $(EXEC) $(SRC) $(LDFLAGS)

# Compile with missed vectorization analysis
opt-vec-missed: CFLAGS += -fopt-info-vec-missed -ftree-vectorizer-verbose=5
opt-vec-missed: $(SRC)
	$(CC) $(CFLAGS) -o $(EXEC) $(SRC) $(LDFLAGS)

# Clean up
clean:
	rm -f $(EXEC) $(OBJ)
