doc:
	cp README.md docs/source/README.md
	rm -r docs/build/plot_directive
	cd docs && $(MAKE) html
