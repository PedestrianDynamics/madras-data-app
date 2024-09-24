## Contact Data Structure

The **GPS** folder contains GPS trajectories in the `.gpx` file format.

The **Contacts.csv** file is structured as follows:

- **Columns**:
  - **Name**: Participant's name.
  - **Date**: Recording date (format: DD/MM/YYYY).
  - **Time-of-stop**: Time when recording stopped (format: HH:MM:SS.mmm).
  - **Total-number-of-collisions**: Number of collisions recorded.
  - **Duration**: Total duration of the recording (format: HH:MM:SS.mmm).
  - **Instant contact**: Time stamps for each contact instance from the start (format: HH:MM:SS.mmm).

- **Rows**:
  - Each row represents a single recording session for a participant.

- **Data Characteristics**:
  - Time data is in the format HH:MM:SS.mmm.
  - Some rows may have missing contact instances, indicated by empty fields.

### Vocabulary

- **Time Format (HH:MM:SS.mmm)**:
  - **HH**: Hours (00-23)
  - **MM**: Minutes (00-59)
  - **SS**: Seconds (00-59)
  - **mmm**: Milliseconds (000-999)

For example, "00:00:33.520" represents 33 seconds and 520 milliseconds.

- **Date Format (DD/MM/YYYY)**:
  - **DD**: Day (01-31)
  - **MM**: Month (01-12)
  - **YYYY**: Year (four digits)

For example, "08/12/2022" represents December 8, 2022.

