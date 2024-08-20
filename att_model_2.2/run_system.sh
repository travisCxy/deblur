cd `dirname $0`

export PYTHONPATH=./:./system_scripts
python3 system_scripts/dispatch.py $*

echo 'run.sh finished'
