# Build Notes


1. CurrentState class tracks current state of the book from market socket updates
2. FairValueModel class takes the current state info and computes fair value of the token to quote around: this is where all the fun is
3. Strategy class takes the state and fairvalue and creates list of desired quotes
4. OrderManager class executes the orders



