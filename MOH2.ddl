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

CREATE TABLE IF NOT EXISTS Hospital(
    hospitalName VARCHAR(100) PRIMARY KEY,
    StreetAdress VARCHAR(100) NOT NULL,
    City VARCHAR(100) NOT NULL,
    AnnualBudget INT NOT NULL
);

CREATE TABLE IF NOT EXISTS Department (
    hospitalName VARCHAR(100),
    Name VARCHAR(100) NOT NULL UNIQUE,
    AnnualBudget INT NOT NULL,
    CONSTRAINT NurseDepartment_FK FOREIGN KEY (hospitalName) REFERENCES Hospital(hospitalName),
    PRIMARY KEY (hospitalName, Name)
);

CREATE TABLE IF NOT EXISTS Nurse(
    uid INT,
    YearlySalaryN INT NOT NULL,
    YearsPracticedN INT NOT NULL,
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

CREATE TABLE IF NOT EXISTS Patient(
    uid INT,
    physician_uid INT,
    nurse_uid INT,
    HealthInsurance VARCHAR(100) NOT NULL,
    CONSTRAINT PatientPerson_FK FOREIGN KEY (uid) REFERENCES Person(uid),
    PRIMARY KEY (uid, HealthInsurance)
);
--this is works
CREATE TABLE IF NOT EXISTS Nurse_Department(
    hospitalName VARCHAR(100),
    nurse_uid INT,
    YearlySalary INT,
    CONSTRAINT Nurse_Department_Department_Fk FOREIGN KEY(hospitalName,DepartmentName) REFERENCES Department(hospitalName, Name),
    CONSTRAINT Nurse_Department_Nurse_fk FOREIGN KEY(uid,YearlySalary) REFERENCES Nurse(uid,YearlySalary)
);

CREATE TABLE IF NOT EXISTS Arrives(
    patient_uid INT, 
    hospitalName VARCHAR(100),
    CONSTRAINT Patient_Arrived_Fk FOREIGN KEY(uid) REFERENCES Patient(uid),
    CONSTRAINT Arrived_hospitalName_Fk FOREIGN KEY(hospitalName) REFERENCES Hospital(hospitalName)
);


CREATE TABLE IF NOT EXISTS AdmissionsRecord(
    AdmitDate VARCHAR(100) NOT NULL,
    Priority VARCHAR(100) NOT NULL,
    patient_uid INT,
    hospitalName VARCHAR(100) NOT NULL,
    CONSTRAINT AdmissionsRecord_Patient_FK FOREIGN KEY(patient_uid) REFERENCES Arrives(patient_uid)
    CONSTRAINT AdmissionsRecord_hospital_Fk FOREIGN KEy(hospitalName) REFERENCES Arrives(hospitalName)
    PRIMARY KEY(patient_uid,hospitalName)
);



CREATE TABLE IF NOT EXISTS GivesDiagnosis (
    patient_uid INT,
    physician_uid INT,
    CONSTRAINT GivesDiagnosis_patient_FK FOREIGN KEY(patient_uid) REFERENCES Patient(uid),
    CONSTRAINT GivesDiagnosis_physician_FK FOREIGN KEY(physician_uid) REFERENCES Patient(physician_uid),
);

CREATE TABLE IF NOT EXISTS Diagnosis (
    Disease VARCHAR(100) NOT NULL,
    Date VARCHAR(100) NOT NULL,
    Prognosis VARCHAR(100) NOT NULL,
    patient_uid INT,
    physician_uid INT,
    CONSTRAINT Diagnosis_patient_FK FOREIGN KEY(patient_uid) REFERENCES GivesDiagnosis(patient_uid),
    CONSTRAINT Diagnosis_physician_FK FOREIGN KEY(physician_uid) REFERENCES GivesDiagnosis(physician_uid)
);



CREATE TABLE IF NOT MedicalTest (
    unid INT PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    fee REAL
);

CREATE TABLE IF NOT Undergoes(
    Test_unid INT,
    Patient_uid INT,
    TestResults VARCHAR(100) NOT NULL,
    TestDate VARCHAR(100) NOT NULL,
    CONSTRAINT MedicalTest_FK FOREIGN(Test_unid) REFERENCES MedicalTest(unid),
    CONSTRAINT Patient_FK FOREIGN(Patient_uid) REFERENCES Patient(uid)
);

CREATE TABLE IF NOT Perscribes(
    Patient_uid INT,
    Physician_uid INT,
    Date VARCHAR(100) NOT NULL,
    CONSTRAINT Diagnosis_patient_FK FOREIGN KEY(Patient_uid) REFERENCES Patient(patient_uid),
    CONSTRAINT Diagnosis_physician_FK FOREIGN KEY(Physician_uid) REFERENCES Patient(physician_uid)
);

CREATE TABLE IF NOT Prscription(
    Drug VARCHAR(100) NOT NULL,
    Dosage VARCHAR(100) NOT NULL,
    Patient_uid INT,
    CONSTRAINT Persciption_patient_FK FOREIGN KEY(Patient_uid) REFERENCES Perscribes(Patient_uid)
);

