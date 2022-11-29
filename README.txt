(a) A list of all external libraries, software, data, or other tools that your
system uses, along with a URL or citation for each one that tells us what it is
and where it came from.

Everything currently being used is:
    numpy - https://numpy.org/ - A library for matrix operations and manipulation
    spacy - https://spacy.io/ - A library for natural language processing

We also have installed the large model for spacy, which can be installed with
`python -m spacy download en_core_web_lg`. We saw explicit permission for usage
of the medium model, so it would be reasonable that the large model is likewise
allowed. If this is a misunderstanding, the medium model can be used instead by
modifying line 36 of `helpers.py` to use `en_core_web_md` instead of the current
`en_core_web_lg`, which gives similar (but not *quite* as good) results.

(b) A time estimate of how long your system takes to process ONE story. (This
is to give the TAs some estimate of how long they will need to wait to see the
results.)

    A story takes 2.01 seconds on average to answer a story--tested on
    39 stories, which took 78.49 seconds.

(c) Please tell us which CADE machine(s) you tested your system on.

    Cade machine: LAB1-23

(d) Any known problems or limitations of your system.

    On some of the questions our program can not find a good match in the
    sentence. This results in the whole sentence being returned as the answer.