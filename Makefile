
DIRS = segment src

dirloop = for dir in $(DIRS); do      \
            echo '  build' $$dir;    \
            $(MAKE) -s -C $$dir $(1); \
          done

all:
	$(call dirloop, )

clean:
	$(call dirloop, clean)
