(a) A list of all external libraries, software, data, or other tools that your
system uses, along with a URL or citation for each one that tells us what it is
and where it came from.

Everything currently being used is:
    numpy - https://numpy.org/ - A library for matrix operations and manipulation
    spacy - https://spacy.io/ - A library for natural language processing

We also have installed the small model for spacy, which can be installed with
`python -m spacy download en_core_web_sm`, and the extra index url installed
for torch, which can be installed with
`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`

(b) A time estimate of how long your system takes to process ONE story. (This
is to give the TAs some estimate of how long they will need to wait to see the
results.)

    An average story takes about 5 second to output.

(c) Please tell us which CADE machine(s) you tested your system on.
 
    Cade machine: LAB1-23

(d) Any known problems or limitations of your system.

    On some of the questions our program can not find a good match in the sentence. This results in the whole sentence being 
    returned as the answer.