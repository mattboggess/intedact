doc:
	cp README.md docs/source/README.md
	rm -rf docs/build/plot_directive
	cd docs && $(MAKE) html
