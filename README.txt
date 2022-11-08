(a) A list of all external libraries, software, data, or other tools that your
system uses, along with a URL or citation for each one that tells us what it is
and where it came from.

Everything currently being used is:
    numpy - https://numpy.org/ - A library for matrix operations and manipulation
    spacy - https://spacy.io/ - A library for natural language processing
    torch - https://pytorch.org/ - A library for neural network training to use Bert
    sklearn - https://scikit-learn.org/stable/ - A library for machine learning algorithms
    transformers - https://huggingface.co/transformers/ - A library to allow us to import Bert


We also have installed the small model for spacy, which can be installed with
`python -m spacy download en_core_web_sm`, and the the extra index url
installed for torch, which can be installed with
`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`

(b) A time estimate of how long your system takes to process ONE story. (This
is to give the TAs some estimate of how long they will need to wait to see the
results.)

An average story takes about 1 second to output on our personal computers.

(c) Any known problems or limitations of your system.

We are currently returning the full sentence that we believe the answer is in
by creating signature vectors of the questions based on the words contained in
the stories. This isn't exactly the *best* approach, but it does seem to give
us roughly 0.18 for the f score.