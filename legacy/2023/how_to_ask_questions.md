# How to effectively ask questions (during the practical sessions and on Canvas)?

## You should never be afraid or feel discouraged to ask questions

The guidelines below are not intended to discourage you from asking questions and guidance during the practical sessions or on Canvas. Quite the opposite, the goal of this short document is to instruct you how to communicate your questions in a more effective way.

## Start your question by explaining the context

During the practical sessions, the student assistants and teachers have to simultaneously answer questions from students working on different exercises (same for Canvas). Before asking the questions, make sure that you properly explain the context. While helpful, simply stating the assignment number (e.g. &quot;Exercise 2.1&quot;) is not sufficient (the student assistants and teachers don&#39;t have the assignments memorized). Good context description should include the goal of the task or assignment that you are working on and the expected outcome or result. If you are asking the question on Canvas, provide a descriptive title for your post so the teachers and other students can immediately get the gist of it.

## Formulate a specific question

&quot;I don&#39;t know how to solve Exercise 2&quot; is not a specific question (and is not even a question). The goal of the practical sessions is not to simply provide you with the answers for the assignments, but to guide you in coming up with an answer yourself.

Always make it clear what you want to get out by asking a question or guidance. If the answer is simply &quot;the solution to the exercise or assignment&quot;, your question is not specific enough.

Please also refrain from asking the student assistants or teachers to check if the complete answer to a question or assignment is correct, particularly for project work. A common question that is asked in this situation is &quot;Is this enough for the project work?&quot;. This is not a specific question, it is strictly not allowed, and is equal to asking for the complete solution to the question or assignment.

## Demonstrate that you have attempted to answer the questions or solve the problem yourself

Before asking for help, you should attempt to answer the question or solve the problem yourself. If you are unsure how to use a certain function, you should look at the documentation for that function. If you do not know which function to use, search the documentation of the development environment that you are using (e.g. MATLAB or Python + numpy). If your code returns an error message, you should try to understand the information that the message relates to you (see below) and look at the documentation of the function that will help you further understand the error. If you do not know how to answer some question, try to formulate a provisional answer (however incorrect you think it might be, you won&#39;t be judged).

When asking for help, always describe your attempts to solve the problem or answer the question.

## Error messages are informative

&quot;It gives an error.&quot; is not a question. Errors are in almost all instances accompanied by error messages that give you information that can help you solve the problem. Look at the following error message for example (this is a MATLAB example, however, error messages look similar in other programming languages as well):

```
Error using *

Inner matrix dimensions must agree.

Error in some_matlab_func (line 6)

C = A*B;
```

It contains several pieces of information. &quot;Error using \*&quot; means that the error occurs when using the multiplication operator &quot;\*&quot;. The next message &quot;Inner matrix dimensions must agree.&quot; means that the dimensions of the matrices that you are trying to multiply are not suitable for performing matrix multiplication (in order to perform matrix multiplication, the number of rows of the first matrix must be equal to the number of columns of the second matrix). Furthermore, the error message says that the error occurs on line 6 of the function `some_matlab_func.m`. This line is `C = A*B;`. Thus, the dimensions of the matrices `A` and `B` that you are trying to multiply are not suitable for performing matrix multiplication. The next step in the debugging process should be to investigate why these dimensions do not match.

## Make sure that the problem is reproducible

It is difficult to debug code that &quot;sometimes works and sometimes doesn&#39;t.&quot; Before asking a question, investigate under which circumstances the error occurs, and under which circumstances the code works as expected (this can be for example, due to different inputs to the function). Make sure that you can always reproduce the error message or unexpected output of your code.

If you ask the question online, make sure that you attach a minimal and standalone example that can be used to reproduce the error. &quot;Minimal&quot;, means that the example contains the minimal number of functions and inputs (such as images) that can be used to reproduce the error or unexpected output. &quot;Standalone&quot; means that the the code includes all functions and inputs that are sufficient to reproduce the error or unexpected output.

## Example

**Bad**: &quot;I am not sure what to do in the first assignment. I tried `Theta = ls solve(addones(trainingX), trainingY);`, but I get an error when I try to plot the model.&quot;

**Good**: &quot;In the first assignment from the exercises from the computer aided diagnosis topic, we need to compute the parameters of a linear regression model for a given dataset. According to me, we need to use the ls_solve function that we previously implemented to solve for the parameters Theta. I do this by running the following line of code `Theta = ls solve(addones(trainingX), trainingY);`. However, when I plot the results with `plot_regression(addones(trainingX), trainingY, Theta);`, I get the following error message:

```
Error using *

Inner matrix dimensions must agree.

Error in plot_regression (line 8)

predictedY = addones(X)*Theta;
```

It seems that the problem occurs when calculating the predicted target values. I assume that it is due to the use of the addones() function.

Attached is my implementation of the `ls_solve.m` function, the `plot_regression.m` function and a `.mat` file containing the trainingX and trainingY variables that can be used to reproduce the problem. &quot;
