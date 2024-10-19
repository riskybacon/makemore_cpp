CPP = g++
EXTRA_CXXFLAGS =
CXXFLAGS = -Wall -O3 --std=c++20 -I. $(EXTRA_CXXFLAGS)

TARGETS = pt1_one_layer

SRCS = pt1_one_layer.cpp
OBJS = $(SRCS:.c=.o)

all: $(TARGETS)

$(TARGET): $(OBJS)
	$(CPP) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CPP) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGETS) *.dSYM
