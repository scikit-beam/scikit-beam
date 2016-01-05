#!/bin/bash
set -e # exit with nonzero exit code if anything fails

make clean
make notebooks
make html


# go to the out directory and create a *new* Git repo
cd _build/html
git init
touch .nojekyll

# inside this git repo we'll pretend to be a new user
git config --global user.name "Eric Dill"
git config --global user.email "edill@bnl.gov"

# The first and only commit to this new Git repo contains all the
# files present with the commit message "Deploy to GitHub Pages".
git add .
git commit -m "Deploy to GitHub Pages"

# add the credentials after **all** the files are added and committed so that
# the OAuth token is not added to the gh-pages branch on github for all to see!
# git config credential.helper "store --file=.git/credentials"
# echo "https://${GH_TOKEN}:@github.com" > .git/credentials

# Force push from the current repo's master branch to the remote
# repo's gh-pages branch. (All previous history on the gh-pages branch
# will be lost, since we are overwriting it.) We redirect any output to
# /dev/null to hide any sensitive credential data that might otherwise be exposed.
git push --force "https://ericdill:${GH_TOKEN}@${GH_REF}" master:gh-pages