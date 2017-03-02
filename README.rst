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

First one need to train a model::

    python manage.py train models/solresol data/solresol-train.conllu

Then one can proceed to parsing some new data using the trained model::

    python manage.py parse models/solresol data/solresol-test.conllu output/solresol.conllu

The ``data``, ``models``, and ``output`` dirs are conveniently git-ignored.


setup
=====

Something like this should do::

    git clone && cd
    venv path/to/envs/mstnn
    source path/to/envs/mstnn/bin/activate
    pip install -r requirements.txt
    python manage.py unittest


licence
=======

MIT. Do as you please and praise the snake gods.
