#!/bin/bash
# Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
echo "Checking python formatting...";

################################################################################
# Python incremental format checker adapted from format-incremental in Cirq.
#
# The base git revision to compare against is chosen from the following defaults,
# in order, until one exists:
#
#     1. upstream/master
#     2. origin/master
#     3. master
#
# If none exists, the script fails.
################################################################################

# Get the working directory to the repo root.
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$(git rev-parse --show-toplevel)"

# Parse arguments.
rev=""
for arg in $@; do
    echo -e "\033[31mNo arguments expected.\033[0m" >&2
    exit 1
done

# Figure out which branch to compare against.
if [ -z "${rev}" ]; then
    if [ "$(git cat-file -t upstream/master 2> /dev/null)" == "commit" ]; then
        rev=upstream/master
    elif [ "$(git cat-file -t origin/master 2> /dev/null)" == "commit" ]; then
        rev=origin/master
    elif [ "$(git cat-file -t master 2> /dev/null)" == "commit" ]; then
        rev=master
    else
        echo -e "\033[31mNo default revision found to compare against.\033[0m" >&2
        exit 1
    fi
fi
base="$(git merge-base ${rev} HEAD)"
if [ "$(git rev-parse ${rev})" == "${base}" ]; then
    echo -e "Comparing against revision '${rev}'." >&2
else
    echo -e "Comparing against revision '${rev}' (merge base ${base})." >&2
    rev="${base}"
fi

# Get the _test version of changed python files.
needed_changes=0
changed_files=$(git diff --name-only ${rev} -- | grep "\.py$" | grep -v "_pb2\.py$")
esc=$(printf '\033')
for changed_file in ${changed_files}; do
    # Extract changed line ranges from diff output.
    changed_line_ranges=$( \
        git diff --unified=0 "${rev}" -- "${changed_file}" \
        | perl -ne 'chomp(); if (/@@ -\d+(,\d+)? \+(\d+)(,)?(\d+)? @@/) {$end=$2+($4 or 1)-1; print "--lines=$2-$end "}' \
    )
    if [[ "${changed_line_ranges}" != "--lines=0-0 " ]]; then
        # Do the formatting.
        results=$(yapf --style=google --diff "${changed_file}" ${changed_line_ranges})

        # Print colorized error messages.
        if [ ! -z "${results}" ]; then
            needed_changes=1
            echo -e "\n\033[31mChanges in ${changed_file} require formatting:\033[0m\n${results}" \
                | sed "s/^\(+ .*\)$/${esc}[32m\\1${esc}[0m/" \
                | sed "s/^\(- .*\)$/${esc}[31m\\1${esc}[0m/"
        fi
     fi
done

if (( needed_changes == 0 )); then
    echo -e "\033[32mNo formatting needed on changed lines\033[0m."
else
    echo -e "\033[31mSome formatting needed on changed lines\033[0m."
    exit 1
fi

echo "Checking C++ formatting...";
formatting_outputs=$(find tensorflow_quantum/ -iname *.h -o -iname *.cc | xargs clang-format -style=google -output-replacements-xml);
CFORMATCHECK=0
while read -r formatting_outputs; do
    if [ "$formatting_outputs" != "<?xml version='1.0'?>" ] && [ "$formatting_outputs" != "<replacements xml:space='preserve' incomplete_format='false'>" ] && [ "$formatting_outputs" != "</replacements>" ]; then
        CFORMATCHECK=64
    fi
done <<< "$formatting_outputs"
if [ "$CFORMATCHECK" == "0" ]; then
    echo "C++ format checking complete!";
    exit 0;
else
    echo "C++ format checking failed, please run the formatting script before proceeding."
    exit 64;
fi
