"""GH Release script for MTLLM."""

from github_release import gh_release_create

import markdown_to_json

import toml


def get_release_info(version: str) -> str:
    """Get release info from CHANGELOG.md."""

    def list_to_markdown(items: list) -> str:
        """Convert list to markdown."""
        return "\n".join([f"- {item}" for item in items])

    with open("docs/changelog.md", "r") as f:
        changelog = f.read()
        changelog_json = markdown_to_json.dictify(changelog)
        for release_str, release_info in changelog_json["RELEASES"].items():
            if version in release_str:
                return list_to_markdown(release_info)
    raise ValueError(f"Version {version} not found in CHANGELOG.md")


with open("pyproject.toml", "r") as f:
    data = toml.load(f)

version = data["tool"]["poetry"]["version"]

gh_release_create(
    "chandralegend/semantix",
    version,
    publish=True,
    name=f"v{version}",
    asset_pattern="dist/*",
    body=get_release_info(version),
)
