-- Table creation DDL:
-- https://dev.mysql.com/doc/refman/8.0/en/integer-types.html
CREATE TABLE IF NOT EXISTS Person (
    uid INT PRIMARY KEY,
    FirstName VARCHAR(100) NOT NULL,
    LastName VARCHAR(100) NOT NULL,
    Gender VARCHAR(20) NOT NULL,
    BirthDate VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS Address (
    uid INT,
    City VARCHAR(100) NOT NULL,
    Street VARCHAR(100) NOT NULL,
    PostalCode VARCHAR(100) NOT NULL,
    Province VARCHAR(100) NOT NULL,
    CONSTRAINT AddressPerson_FK FOREIGN KEY (uid) REFERENCES Person(uid),
    PRIMARY KEY(uid, PostalCode)
);

CREATE TABLE IF NOT EXISTS PhoneNumber(
    uid INT,
    Number VARCHAR(100) NOT NULL UNIQUE,
    ContactType VARCHAR(100) NOT NULL,
    CONSTRAINT PhoneNumberPerson_FK FOREIGN KEY (uid) REFERENCES Person(uid),
    PRIMARY KEY(uid, Number)
);

CREATE TABLE IF NOT EXISTS Patient(
    uid INT,
    HealthInsurance VARCHAR(100) NOT NULL,
    CONSTRAINT PatientPerson_FK FOREIGN KEY (uid) REFERENCES Person(uid),
    PRIMARY KEY (uid, HealthInsurance)
);

CREATE TABLE IF NOT EXISTS Hospital(
    hospitalName VARCHAR(100) PRIMARY KEY,
    StreetAdress VARCHAR(100) NOT NULL,
    City VARCHAR(100) NOT NULL,
    AnnualBudget INT NOT NULL,
);

CREATE TABLE IF NOT EXISTS Department (
    hospitalName VARCHAR(100),
    Name VARCHAR(100) NOT NULL UNIQUE,
    AnnualBudget INT NOT NULL,
    CONSTRAINT NursePerson_FK FOREIGN KEY (hospitalName) REFERENCES Hospital(hospitalName),
    PRIMARY KEY (hospitalName, Name)
);

CREATE TABLE IF NOT EXISTS Nurse(
    uid INT,
    YearlySalary INT NOT NULL,
    YearsPracticed INT NOT NULL,
    CONSTRAINT NursePerson_FK FOREIGN KEY (uid) REFERENCES Person(uid),
    PRIMARY KEY(uid, YearlySalary)
);

CREATE TABLE IF NOT EXISTS Physician(
    uid INT,
    YearsPracticed INT NOT NULL,
    YearlySalary INT NOT NULL,
    MedicalSpecialty VARCHAR(100) NOT NULL,
    DepartmentName VARCHAR(100) NOT NULL,
    CONSTRAINT PhysicianPerson_FK FOREIGN KEY (uid) REFERENCES Person(uid),
    CONSTRAINT DepartmentName_FK FOREIGN KEY (DepartmentName) REFERENCES Department(Name),
    PRIMARY KEY (uid, MedicalSpecialty)
);




