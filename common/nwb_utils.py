from pynwb import NWBFile
from pynwb.file import Subject
from uuid import uuid4


def create_base_nwb_file(nwbfile_original: NWBFile) -> NWBFile:
    return NWBFile(
        session_description=nwbfile_original.session_description + " - spike sorting results.",
        identifier=str(uuid4()),
        session_start_time=nwbfile_original.session_start_time,
        session_id=nwbfile_original.session_id,
        experimenter=nwbfile_original.experimenter,
        lab=nwbfile_original.lab,
        institution=nwbfile_original.institution,
        experiment_description=nwbfile_original.experiment_description,
        keywords=nwbfile_original.keywords,
        notes=nwbfile_original.notes,
        pharmacology=nwbfile_original.pharmacology,
        protocol=nwbfile_original.protocol,
        related_publications=nwbfile_original.related_publications,
        data_collection=nwbfile_original.data_collection,
        surgery=nwbfile_original.surgery,
        virus=nwbfile_original.virus,
        subject=Subject(
            age=nwbfile_original.subject.age,
            description=nwbfile_original.subject.description,
            genotype=nwbfile_original.subject.genotype,
            sex=nwbfile_original.subject.sex,
            species=nwbfile_original.subject.species,
            subject_id=nwbfile_original.subject.subject_id,
            weight=nwbfile_original.subject.weight,
            date_of_birth=nwbfile_original.subject.date_of_birth,
            strain=nwbfile_original.subject.strain,
        )
    )
