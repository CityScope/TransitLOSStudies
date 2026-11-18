"""Phase staggering of frequency-based feeds. The implementation moved to `pyGTFSHandler.downloaders.utils.fleet_frequency.stagger_feed`
(documented in pyGTFSHandler/docs/methodology/new_area_gtfs_agent_guide.md, section 8); this module keeps the old import and command line working.

    python stagger_phase.py <feed_dir> [<feed_dir> ...]
"""
import sys

from pyGTFSHandler.downloaders.utils.fleet_frequency import stagger_feed  # noqa: F401

if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, stagger_feed(f))
