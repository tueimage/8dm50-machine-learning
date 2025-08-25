
# Machine Learning in Medical Imaging and Biology (8DM50)
The course covers a number of machine learning methods and concepts, including state-of-the-art deep learning methods, with example applications in the medical imaging and computational biology domains.

## Use of Canvas
This GitHub page contains all the general information about the course and the study materials. The [Canvas page of the course](https://canvas.tue.nl/courses/29625) will be used only for sharing of course information that cannot be made public (e.g. Microsoft Teams links), submission of the practical work and posting questions to the instructors and teaching assistants (in the Discussion section). The students are highly encouraged to use the Discussion section in Canvas. All general questions (e.g. issues with setting up the programming environment, error messages etc., general methodology questions) should be posted in the Discussion section.

**TLDR**: GitHub is for content, Canvas for communication and submission of assignments.

## Schedule

The course schedule is as follows:
* **Lectures**, *time*: Wednesdays 08:45 - 10:45, *location*: Gemini-zuid 3A.06 or Auditorium 14 (check your timetable for details).
* **Guided self-study**, *time*: Wednesdays 10.45 - 12.45, *location*: Metaforum zaal 08 or Metaforum zaal 07 (check your timetable for details).

## Practical work

The practical work will be done in groups. The groups will be formed in Canvas and you will also submit all your work there (check the Assignments section for the deadlines). Your are expected to do this work independently with the help of the teaching assistants during the guided self-study sessions (*begeleide zelfstudie*). You can also post your questions in the Discussion section in Canvas at any time (i.e. not just during the practical sessions).

**IMPORTANT: Please read [this guide](how_to_ask_questions.md) on effectively asking questions during the practical sessions.**

### Use of ChatGPT and other large language models

The use of ChatGPT and other large language models for the practical work is allowed, provided that:

1) You use ChatGPT and other large language models only as aid in your work and not as primary sources of information (e.g. to do literature search), and primary mode of writing and coding (e.g. asking for answers to entire assignment questions is not allowed, however, improving the writing or coding of answers to questions is allowed).
   
2) You write a one-page reflection report on the use of such tools answering the following questions:
    * What were the up- and down-sides of using ChatGPT (or similar tools) in your work?
    * In your view, are such tools helpful or harmful when used in education?
    * Did it make you more or less productive?
    * How did you specifically use these tools (give examples)?
    * Were these tools accurate in their answers?

Note that the report is **mandatory** if you used ChatGPT (or similar tools) in any way for the course and it does NOT have any negative consequence (e.g. lead to lower grades). If you do not submit the report we will assume that you did not use such tools but if this is detected during the grading it will be considered cheating. 

# Materials

## Books
The lectures are mainly based on the selected chapters from the following two books that are freely available online:

* [Deep Learning](https://www.deeplearningbook.org/), Ian Goodfellow and Yoshua Bengio and Aaron Courville
* [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/), Trevor Hastie, Robert Tibshirani, Jerome Friedman

The follwing book is optional but highly recommended:
* [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html), Kevin Patrick Murphy.

Additional reading materials such as journal articles are listed within the lecture slides.

## Software

**IMPORTANT: It is essential that you correctly set up the Python working environment by the end of the first week of the course so there are no delays in the work on the practicals.**

The practical assignments for this course will be done in Python. Please carefully follow [the instructions available here](software.md) on setting up the working environment and (optionally) a Git workflow for your group.

## Python quiz

**IMPORTANT: Attempting the quiz before the specified deadline is mandatory.**

In the first week of the course you have to do a Python self-assessment quiz in Canvas. The quiz will not be graded. If you fail to complete the quiz before the deadline, you will not get a grade for the course. The goal of the quiz is to give you an idea of the Python programming level that is expected.

If you lack prior knowledge of the Python programming language, you can use the material in the "Python essentials" and "Numerical and scientific computing in Python" modules available [here](https://github.com/tueimage/essential-skills/).

## Lectures and assignments

The course is largely unchanged from the previous edition and the materials (slides and practicals) will remain similar. In the schedule below there are tentative materials copied from the previous year so you can have a sneak peek. The tentative materials are indicated with ~~strikethrough~~ and will be updated as the course progresses.

### Lectures

| # | Date | Title | Slides |
| --- | --- | --- | --- |
| 1 | 04/Sep | Machine learning fundamentals | [intro ](lectures/intro.pdf), [slides](lectures/week_1.pdf), [extended](lectures/week_1_extended.pdf) |
| 2 | 11/Sep | Linear models | [slides](lectures/week_2.pdf) |
| 3 | 18/Sep | Deep learning I | [slides](lectures/week_3.pdf)  |
| 4 | 25/Sep | Deep learning II | [slides](lectures/week_4.pdf)|
| 5 | 02/Oct | Support vector machines, random forests | [slides](lectures/week_5.pdf) | 
| 6 | 09/Oct | Unsupervised machine learning | [slides](lectures/week_6.pdf) |
| 7 | 16/Oct | Transformers, ~~Explainable AI~~| [slides](lectures/week_7.pdf), ~~[explainable AI slides](lectures/FGrisoni_2023_Guest_XAI.pdf)~~|
| 8 | 23/Oct | Explainable AI (guest lecture) | [explainable AI slides](lectures/FGrisoni_2023_Guest_XAI.pdf) |
| :small_red_triangle:| 30/Oct | *Exam* | [Example exam](exam.pdf) |

### Practical assignments

| # | Date | Title | Exercises |
| --- | --- | --- | --- |
| 1 | 06/Sep | Machine learning fundamentals I| [exercises](practicals/week_1.ipynb) |
| 2 | 11/Sep | Machine learning fundamentals II|  [exercises](practicals/week_2.ipynb)  |
| 3 | 18/Sep | Linear models |  [exercises](practicals/week_3.ipynb)  |
| 4 | 25/Sep | Deep learning I | [exercises](practicals/week_4.ipynb) |
| 5 | 02/Oct | Deep learning II  | [updated exercises in Google Colab](https://colab.research.google.com/drive/1O-tHiagXjYTXB5Aic92FR9zawJzmZZx9?usp=sharing) |
| 6 | 09/Oct | Support vector machines, random forests | [exercises](practicals/week_6.ipynb) |
| 7 | 16/Oct | *Catch up week!* :tomato:  | - |

# Other course information

## Learning objectives

After completing the course, the student will be able to:
* Recognise how machine learning methods can be used to solve problems in Medical Imaging and Computational Biology.
* Comprehend the basic principles of machine learning.
* Implement and use machine learning methods.
* Design experimental setups for training and evaluation of machine learning models.
* Analyze and critically evaluate the results of experiments with machine learning models.

## Assessment

The assessment will be performed in the following way:

* Work on the practical assignments: 25% of the final grade (each assignment has equal contribution);
* Reading assignment: 10% of the final grade;
* Final exam: 65% of the final grade.

Intermediate feedback will be provided as grades to the submitted assignments.

The grading of the assignments will be done per groups, however, it is possible that individual students get separate grade from the rest of the group (e.g. if they did not sufficiently participate in the work of the group).

An example exam can be found [here](exam.pdf).

## Instruction

The students will receive instruction in the following ways:

* Lectures
* Guided practical sessions with the teaching assistants for questions, assistance and advice
* On-line discussion

Course instructors:
* Mitko Veta
* Federica Eduati

Teaching assistants:
* Hong Liu 
* Hassan Keshvarikhojasteh
* Glen Weber

## Recommended prerequisite courses

8DB00 Image acquisition and Processing, and 8DC00 Medical Image Analysis.



*This page is carefully filled with all necessary information about the course. When unexpected differences occur between this page and Osiris, the information provided in Osiris is leading.*
