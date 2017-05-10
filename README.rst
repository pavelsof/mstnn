=====
mstnn
=====

| “Here lies a toppled god.
| His fall was not a small one.
| We did but build his pedestal,
| A narrow and a tall one.”
| 
| Tleilaxu Epigram


usage
=====

First one needs to train a model::

    python manage.py train models/solresol data/solresol-train.conllu

Training is configurable: e.g. one can specify a development dataset to check
the uas score against at the end of each training epoch; or specify a
pre-trained lemma embeddings file; or exclude lemmas from the feature set
altogether. Invoking ``python manage.py train --help`` lists the options.

Then one can proceed to parsing some fresh data using the trained model::

    python manage.py parse models/solresol data/solresol-test.conllu output/solresol.conllu

The ``data``, ``models``, and ``output`` dirs are conveniently git-ignored.

There are a couple of other cli commands as well, ``python manage.py --help``
lists these.


setup
=====

Something like this should do::

    git clone && cd
    venv path/to/envs/mstnn
    source path/to/envs/mstnn/bin/activate
    pip install -r requirements.txt
    python manage.py unittest

The neural network is built entirely with `Keras`_ and the latter's backend
should not matter.


idea
====

If you are here accidentally, but you are still here nonetheless: this is a
graph-based dependency parser, a descendant of sorts of `MSTParser`_. It uses a
neural network to predict edge probabilities that are then fed into an
implementation of the cool `Chu–Liu/Edmond's algorithm`_ in order to produce
the most probable parse tree. It only works with `CoNLL-U`_ datasets but making
it read other formats would be easy.


licence
=======

MIT. Do as you please and praise the snake gods.


.. _`Keras`: https://keras.io/
.. _`MSTParser`: http://www.seas.upenn.edu/~strctlrn/MSTParser/MSTParser.html
.. _`Chu–Liu/Edmond's algorithm`: https://en.wikipedia.org/wiki/Edmonds'_algorithm
.. _`CoNLL-U`: http://universaldependencies.org/format.html
