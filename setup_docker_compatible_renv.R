#!/usr/bin/env Rscript

# Docker-compatible R package installation using renv
# Use renv for version control, then copy to site-library for multi-stage builds

cat("Setting up R environment using renv for Docker compatibility...\n")

# Install renv if not available
if (!require("renv", quietly = TRUE)) {
    install.packages("renv", repos = "https://cran.r-project.org")
}

# Initialize renv in current directory if renv.lock exists
if (file.exists("renv.lock")) {
    cat("Found renv.lock, initializing renv...\n")
    renv::init(bare = TRUE)

    # Restore packages from renv.lock
    cat("Restoring packages from renv.lock...\n")
    renv::restore(confirm = FALSE)

    cat("✓ renv restore completed\n")
} else {
    cat("No renv.lock found, setting up fresh renv environment...\n")
    renv::init(bare = TRUE)

    # Install essential packages
    essential_packages <- c("BiocManager", "rjson", "GSVA", "remotes", "AUCell")

    cat("Installing essential packages via renv...\n")
    renv::install("BiocManager")
    renv::install("rjson")
    renv::install("bioc::GSVA")
    renv::install("remotes")
    renv::install("bioc::AUCell")

    # Install dpGMM from GitHub
    cat("Installing dpGMM from GitHub...\n")
    renv::install("ZAEDPolSl/dpGMM")

    # Create snapshot
    cat("Creating renv snapshot...\n")
    renv::snapshot()
}

# Now copy packages from renv library to site-library for Docker multi-stage
cat("\nCopying packages from renv to site-library...\n")

# Find renv library path
renv_lib <- renv::paths$library()
cat("renv library path:", renv_lib, "\n")

# Create site-library path
site_lib <- "/usr/local/lib/R/site-library"
dir.create(site_lib, recursive = TRUE, showWarnings = FALSE)

# Get list of installed packages in renv
installed_pkgs <- installed.packages(lib.loc = renv_lib)
pkg_names <- rownames(installed_pkgs)

cat("Copying", length(pkg_names), "packages to site-library...\n")

# Copy each package
copy_count <- 0
for (pkg in pkg_names) {
    src_path <- file.path(renv_lib, pkg)
    dst_path <- file.path(site_lib, pkg)

    if (dir.exists(src_path)) {
        # Check if it's a symlink and resolve it
        if (Sys.readlink(src_path) != "") {
            # It's a symlink, follow it to get the actual path
            actual_path <- Sys.readlink(src_path)
            if (file.exists(actual_path)) {
                # Copy from the actual path, not the symlink
                result <- system2("cp", args = c("-rL", actual_path, dst_path), stdout = FALSE, stderr = FALSE)
                if (result == 0) {
                    copy_count <- copy_count + 1
                    cat("  Copied", pkg, "from", actual_path, "\n")
                } else {
                    cat("Warning: Failed to copy", pkg, "from", actual_path, "\n")
                }
            } else {
                cat("Warning: Symlink target does not exist for", pkg, "->", actual_path, "\n")
            }
        } else {
            # It's a regular directory, copy normally
            result <- system2("cp", args = c("-r", src_path, dst_path), stdout = FALSE, stderr = FALSE)
            if (result == 0) {
                copy_count <- copy_count + 1
                cat("  Copied", pkg, "(regular directory)\n")
            } else {
                cat("Warning: Failed to copy", pkg, "\n")
            }
        }
    }
}

cat("✓ Copied", copy_count, "packages to site-library\n")

# Update library paths to include site-library
.libPaths(c(site_lib, .libPaths()))
cat("Updated library paths:", .libPaths(), "\n")

# Verify essential packages are available
essential_packages <- c("BiocManager", "rjson", "GSVA", "dpGMM", "AUCell")
cat("\nVerifying essential packages:\n")
for (pkg in essential_packages) {
    if (require(pkg, quietly = TRUE, character.only = TRUE)) {
        version_info <- tryCatch(
            {
                as.character(packageVersion(pkg))
            },
            error = function(e) "unknown"
        )
        cat(sprintf("  ✓ %s (%s)\n", pkg, version_info))
    } else {
        cat(sprintf("  ✗ %s (failed)\n", pkg))
    }
}

# List contents of site-library
cat("\nContents of site-library:\n")
if (dir.exists(site_lib)) {
    contents <- list.dirs(site_lib, full.names = FALSE, recursive = FALSE)
    for (item in sort(contents)) {
        if (item != "") {
            cat(sprintf("  - %s\n", item))
        }
    }
}

cat("\nrenv packages successfully copied to site-library for Docker multi-stage build!\n")
