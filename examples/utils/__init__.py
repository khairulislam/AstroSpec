"""Shared loading, preprocessing, and pretrained-model helpers for the examples.

Every notebook imports from this package, so loading a local ``.env`` here,
once, is what actually reaches every notebook -- unlike calling
``load_dotenv()`` in each notebook's own imports cell, which is easy to add to
some and forget on the next one.
"""

from dotenv import load_dotenv

load_dotenv()
