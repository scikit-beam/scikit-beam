.. _git-guide:

Introduction to Git
===================

===============
Configuring git
===============

There are many resources on the web for how to use and configure
`git`, this is a streamlined reference for use at NSLS-II.  There
is a variety of gui front ends for `git`, this will demonstrate the
command line usage as the lowest-common-denominator.  On windows and
OSX the github applications seems pretty slick and integrates well with
the website.

To make pushing/pulling from github as frictionless as possible, it
is best to generate an ssh key pair which will allow see
https://help.github.com/articles/generating-ssh-keys for a
guide of how to generate and upload the keys to github.  If you are
working on a computer on the controls network ssh will not work.


The global behavior of `git` in controlled through `~/.gitconfig`
(`C:\Users\MyLogin\.gitconfig` on Windows), which can edited either
directly, or through the command line (`git config --global ...`).
This is as sugested global configuration file ::

   [user]
	name = Your Name
	email = user@bnl.gov
   [color]
   	diff = auto
   	status = auto
   	branch = auto
   [push]
   	default = upstream

These settings can be over-ridden on a per-repository basis (for
example if you contribute to other projects under a different
email).


====================
Forking and  Cloning
====================

The first step to with the code in github is to create a personal
copy of the repository on github.  This is accomplished by
navigating to the correct repository in https://github.com/NSLS-II/
while logged into github and clicking the **fork** button in the upper
right hand corner.  Follow the on-screen and you should now a copy of
the repository attached to your account on github.


In order
to work on the code you need to get a copy onto your local machine(s) which is
done via **cloning** your github repository to your machine ::

   mkdir ~/my_source  # Any directory will do
   cd ~/my_source     # but this is the scheme I use
   git clone git@github.com:github_user_name/repository_name.git

where `github_user_name` is replaced with your user name.  You may be
asked to unlock your sshkey or authenticate in some way and to accept
the RSA key of the remote host (say yes).  Once you have done this you
should have a directory in your current directory which contains the
most up-to-date version of the code.


========================
Branching and committing
========================

`git` stores the history of code as a series of 'snapshots' of the
project, **commit**\ s, in a tree.  Each **commit** has a unique
identifier generated from it's contents.  A **branch** points to a
single commit and provides a human-readable label for a 'snapshot' in
the project history.

Branching
---------

On a fresh checkout there will be one **branch**, *master*.  To see
a list of the branches currently in your repository run ::

   git branch

which should return ::

   * master

In general, it is a bad idea to work directly on the *master* branch
(for social, not technical reasons), so we will create a new branch to
work on ::

   git branch new_feature

where 'new_feature' is the name of your branch.  The branch name should
be chosen to be descriptive of the type of work you plan to do.  For example
'add_bin1D_function' or 'fix_bug_in_bin1D'.
To see the results of this run ::

   git branch

again which should now print out ::

   * master
     new_feature

There are two things that are important to note here: first we have
not changed anything about the project, just created a new label for
a commit; second we are still 'on *master* '.  To switch to our new
branch we need to 'check it out' via ::

   git checkout new_feature

Running ::

   git branch

again should print ::

     master
   * new_feature

confirming that we have switched branches.  We are now ready
to start working.


.. note:: The process of creating and switching to a new branch can be
    done in a single commend with ::

       git checkout -b new_feature


Editing, Staging, Committing
----------------------------
Now that we are on a new branch we are ready to start working.  You can
use what ever editor you want to create and edit files.  When you reach a
point in your work when you want to save what you have done (ideally this
should be a minimal self-contained change) it is time to **commit** your
work to git.  The first thing you should do is run ::

  git status

which in the case of working on this text prints ::

    # On branch add_sphinx
    # Changes not staged for commit:
    #   (use "git add <file>..." to update what will be committed)
    #   (use "git checkout -- <file>..." to discard changes in working directory)
    #
    #       modified:   source/dev_guide/git_guide.rst
    #       modified:   source/index.rst
    #
    # Untracked files:
    #   (use "git add <file>..." to include in what will be committed)
    #
    #       source/dev_guide/index.rst
    no changes added to commit (use "git add" and/or "git commit -a")

which shows there are two files that have been changed and one new
file created, and no files added to the **index** sense the last
**commit**.  This message also gives some helpful advice on how to
proceed.  To add files to the commit use **add**  ::

    git add filename1, filename2, ...

You can also use shell expansions.  After **add**\ ing the files we
want to **commit**, running **status** again prints::

    # On branch add_sphinx
    # Changes to be committed:
    #   (use "git reset HEAD <file>..." to unstage)
    #
    #       modified:   source/dev_guide/git_guide.rst
    #       new file:   source/dev_guide/index.rst
    #       modified:   source/index.rst
    #

Having confirmed that things look right (you didn't miss any files or
add files that should not be committed) run ::

   git commit

which will open a text editor and prompt you enter a message to go
with your commit.  The message should start with a one-line summary of
the change and then a few sentences describing the changes in more
detail.  The commit message for this commit will be ::

   DOC : basic git usage

   Added text about basic git usage.

Repeat this process as often as necessary.

Changing Branches
-----------------

The files in the repository directory are what `git` refers to as you
**working copy**.  When you switch **branch**\ es `git` will make your
working copy look exactly like the snapshot saved in the **commit**
the **branch** points to.  For example say you are working on
**branch** *new_feature* and you notice an un-related bug.  You should
**commit** all of your feature work (or use **stash**) and then switch back
to the `master` branch ::

  git checkout master

The **working copy** now contains none of your new work.  Create a new
branch to fix the bug ::

  git checkout -b bug_fix

Once you have fixed and committed the bug, switch back to your feature
branch::

  git checkout new_feature

and pick up where you left off.

Reverting edits
---------------

Sometimes edits just are not working out and you need to throw away
all uncommitted changes to a file or the entire **working copy**.  For
a single file ::

   git checkout -- file_name

(the space between `--` and `file_name` is important) will reset the
file to what it looks like on the current branch.  To throw out
*all* of your changes and reset your working directory to the last
commit on your branch ::

   git reset --hard current_branch

=============
Collaborating
=============


Remotes
-------

One of the powerful ideas of distributed version control is that all
clones of a repository are *technically* equivalent.  However, for
organizational reasons we designate one to be the 'canonical'
repository, in this case the repository associated with the
``scikit-beam`` group on github.

In order to get the lastest code from github to your local machine you
need to tell ``git`` where the other code is.  These locations are, in
the language of ``git``, **remotes**.  The first remote we will want to
add in the canonical repository::

    # make sure you are in the working directory of your local repo
    cd ~/<<my_source>>/scikit-beam-examples
    # add the canonical repo as 'upstream'
    git remote add upstream git@github.com:scikit-beam/scikit-beam-examples.git
    # fetch the commits in the new repository
    git fetch upstream

To checkout your handy work run ::

   git remote -v

which should print something like: ::

    origin  git@github.com:username/repo_name.git (fetch)
    origin  git@github.com:username/repo_name.git (push)
    upstream        git@github.com:scikit-beam/scikit-beam-examples.git (fetch)
    upstream        git@github.com:scikit-beam/scikit-beam-examples.git (push)


which shows two remotes (origin and upstream).  It is recommended to
re-name ``origin`` -> your github username ::


   git remote rename origin gh_username

which is the convention that will be used throughout.  You can also
add as a remote the github repositories of other group members, ex ::

   git remote add tacaswell git@github.com:tacaswell/scikit-beam-examples.git

which will allow you to **fetch** to your local computer any commits
they have **push**\ ed to github.


Fetch
^^^^^

Fetching is very simple, assuming you have added the repository you want
to **fetch** from as a **remote** ::

   get fetch remote_name

which will copy all of the commits in the **remote** repository that
are not already in your local repository.   This does not change your
**working copy**, only updates what **commit**\ s `git` knows about.

To checkout a local copy of a remote **branch** ::

   git checkout -t remote_name/remote_branch

See :ref:`git-merging` for how to merge these changes into your branches.

Push
^^^^
**push** is the symmetric operation to **fetch** as ships commits *to*
a remote.   The first time you **push** a **branch** you need to tell `git`
which branch on the **remote** to push to::

   git push --set-upstream a_remote branch_name

and all subsequent times you can just use ::

   git push a_remote

This is the mechanism to share code with in the group, as once you
have pushed **commit**\ s to github, anyone who can see your repository
can **fetch** them and begin to work with them.

.. _git-merging:


Merging
^^^^^^^

You merge two branches by changing to the branch you would like to merge *into* and running ::

   git merge merge_source

If your current branch has no commits that are not in *merge_source*
it is called a 'fast-forward' merge and will always succeed.  If your
local branch has commits that are not in `merge_source` the merge can
generate conflicts which will need to resolved by hand.

Pull
^^^^

The steps of **fetch** and **merge** can be done in one step via ::

   git pull remote_name



===================
Rebase on to master
===================

TODO, see mpl or numpy doc or ask google/stackoverflow

====
Help
====

If you have any issues contact Thomas Caswell tcaswell@bnl.gov ex3146
