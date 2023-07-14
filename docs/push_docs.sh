#!/bin/bash
set -e # exit with nonzero exit code if anything fails

git config --global user.email "Travis@nomail"
git config --global user.name "Travis"
git config --global push.default simple

# go to the out directory and create a *new* Git repo
cd _build/html
git init
touch .nojekyll

# The first and only commit to this new Git repo contains all the
# files present with the commit message "Deploy to GitHub Pages".
git add .
git commit -m "Deploy to GitHub Pages"

git remote add origin git@github.com:scikit-beam/scikit-beam.git

git push origin master:gh-pages --force
