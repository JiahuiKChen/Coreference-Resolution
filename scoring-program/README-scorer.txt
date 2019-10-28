The scoring program can be run as follows:

   python3 scorer.py <KeyDir> <ResponseDir> <Ids> [ -v ]

where:

KeyDir is a directory containing answer key files

ResponseDir is a directory containing response files

Ids is a text file listing the ids of the stories to be scored
  (the answer keys and response files for these stories should be in
  the previously specified directories)

-v is an optional argument for Verbose printing. Without this flag,
 the scorer only shows information for the incorrect and missing
 referents. With the -v flag, the scorer shows information for the
 correct referents as well as the incorrect and missing referents.

==================================================================

You can play with the scoring program using the provided sample files
using the following commands:

   python3 scorer.py keys/ responses/ test1.txt -v
   python3 scorer.py keys/ responses/ test2.txt -v
   python3 scorer.py keys/ responses/ both.txt -v


