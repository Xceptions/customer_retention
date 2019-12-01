### A Machine Learning Project built on Customer Retention

This project was built to predict the customer retention capability of a bank after 2 years.

#### Note
This analysis was done on randomized data that falls within / close to the range of the actual data in the project and not the actual data itself. This was done purposely to ensure that actual company data is protected and company privacy respected. Also, all columns were renamed to dummy names of the alphabet while the target column was named "target". The main aim of this project is to show the process taken in our machine learning.

All columns are numbers

1. 1 - Customer retained,
2. 0 - Customer lost

**To run :** 
You have to build the database, and intialize it first before running the server.
This means you have to first run

```docker-compose up postgres```

Then you run
```docker-compose up initdb```

After the initdb process has run successfully, you then comment it out, and with subsequent runs, you can
just do:
```docker-compose up```