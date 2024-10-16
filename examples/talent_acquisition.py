"""AI Talent Acquisiton Helper"""

from dataclasses import dataclass
from typing import List, Optional

from pypdf import PdfReader

import semantix as sx

llm = sx.llms.OpenAI()


def read_pdf(file_path: str) -> str:
    """Read the content of a PDF file"""
    reader = PdfReader(file_path)
    full_text = []
    for page in reader.pages:
        full_text.append(page.extract_text())
    return "\n".join(full_text)


@dataclass
class Education:
    school_name: str
    degree: str
    field_of_study: str
    start_date: str
    end_date: str
    activities: str
    grade: str
    additional_info: Optional[str] = None


@dataclass
class WorkExperience:
    title: str
    company: str
    location: str
    location_type: sx.Semantic[str, "Onsite | Hybrid | Remote"]
    start_date: str
    end_date: str
    description: str
    additional_info: Optional[str] = None


@dataclass
class Skill:
    name: str
    level: str
    additional_info: Optional[str] = None


@dataclass
class Project:
    name: str
    start_date: str
    end_date: str
    description: str
    url: str
    additional_info: Optional[str] = None


@dataclass
class Certification:
    name: str
    issuing_organization: str
    issue_date: str
    expiration_date: str
    credential_id: str
    additional_info: Optional[str] = None


@dataclass
class Publication:
    title: str
    authors: List[str]
    publication_date: str
    publisher: str
    description: str
    additional_info: Optional[str] = None


@dataclass
class Reference:
    name: str
    position: str
    company: str
    email: str
    phone: str
    additional_info: Optional[str] = None


@dataclass
class Profile:
    name: str
    email: str
    phone: str
    address: str
    summary: str
    education: List[Education]
    experience: List[WorkExperience]
    skills: List[Skill]
    projects: List[Project]
    certifications: List[Certification]
    publications: List[Publication]
    references: List[Reference]
    links: List[str] = None
    additional_info: List[str] = None


@llm.enhance()
def extract_profile(resume_content: str) -> Profile: ...


@dataclass
class JobDescription:
    title: str
    company: str
    location: str
    description: str
    requirements: List[str]
    responsibilities: List[str]
    additional_info: Optional[str] = None


@dataclass
class Evaluation:
    summary: str
    education_match: float
    experience_match: float
    skills_match: float
    overall_score: float


@llm.enhance(method="Reason")
def evaluate_candidate(
    profile: Profile, job_description: JobDescription
) -> Evaluation: ...


if __name__ == "__main__":
    profile_text = read_pdf("examples/resume.pdf")
    profile = extract_profile(resume_content=profile_text)
    job_desc = JobDescription(
        title="Data Scientist",
        company="ABC Company",
        location="Remote",
        description="We are looking for a Data Scientist to join our team.....",
        requirements=["Python", "Machine Learning", "Deep Learning"],
        responsibilities=["Data Analysis", "Model Building", "Data Visualization"],
    )
    evaluation = evaluate_candidate(profile=profile, job_description=job_desc)
    print(profile, evaluation)
