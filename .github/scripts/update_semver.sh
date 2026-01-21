#!/bin/bash

VERSION=$1
PREFIX=$2

IFS='.' read -r -a semver <<< "$VERSION"

if [ -z "${semver[0]}" ] || [ -z "${semver[1]}" ] || [ -z "${semver[2]}" ]; then
    echo "::error:: Invalid version format. Expected x.y.z"
    exit 1
fi

full_version="${VERSION}"
major_version="${semver[0]}"
minor_version="${semver[0]}.${semver[1]}"
if [ -n "$PREFIX" ]; then
    full_version="${PREFIX}${VERSION}"
    major_version="${PREFIX}${major_version}"
    minor_version="${PREFIX}${minor_version}"
fi

if git rev-parse "$full_version" >/dev/null 2>&1; then
    echo "::notice::Version '$full_version' already exists. Skipping tag creation."
else
    # Configure git for pushing tags
    git config user.name "github-actions[bot]"
    git config user.email "github-actions[bot]@users.noreply.github.com"

    git tag "$full_version"
    git tag -f "$major_version"
    git tag -f "$minor_version"
    git push --tags --force

    echo "tag=$full_version" >> $GITHUB_OUTPUT
    echo "minor_tag=$minor_version" >> $GITHUB_OUTPUT
    echo "major_tag=$major_version" >> $GITHUB_OUTPUT

    #output version tag
    echo "version_tag=${full_version}" >> "$GITHUB_OUTPUT"
fi
