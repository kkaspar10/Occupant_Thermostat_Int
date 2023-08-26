#!/bin/sh

# python simulation/energyplus.py select_buildings
python simulation/energyplus.py simulate
python simulation/energyplus.py build_schema
