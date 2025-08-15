"""
Process-based R execution for Streamlit compatibility.

This module provides a process-isolated way to run R code, avoiding the
threading issues that plague rpy2 in Streamlit environments.
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RProcessError(Exception):
    """Raised when R process execution fails."""

    pass


class RProcessExecutor:
    """Execute R code in isolated processes to avoid threading issues."""

    def __init__(self, r_libs_path: Optional[str] = None):
        """
        Initialize R process executor.

        Args:
            r_libs_path: Optional path to R libraries directory
        """
        # Detect renv environment
        renv_activate = Path("renv/activate.R")
        if renv_activate.exists():
            # Use renv library path
            # renv library is usually at renv/library/<platform>/<R-version>
            # We'll try to find the first subdir under renv/library
            renv_lib_root = Path("renv/library")
            renv_lib_path = None
            if renv_lib_root.exists():
                for subdir in renv_lib_root.iterdir():
                    if subdir.is_dir():
                        # Use the first platform subdir (e.g., 'macos', 'linux', etc.)
                        for rver in subdir.iterdir():
                            if rver.is_dir():
                                renv_lib_path = str(rver)
                                break
                        if renv_lib_path:
                            break
            self.using_renv = True
            self.renv_lib_path = renv_lib_path
            self.r_libs_path = (
                renv_lib_path
                or r_libs_path
                or os.environ.get("R_LIBS_USER", "/usr/local/lib/R/site-library")
            )
        else:
            self.using_renv = False
            self.renv_lib_path = None
            self.r_libs_path = r_libs_path or os.environ.get(
                "R_LIBS_USER", "/usr/local/lib/R/site-library"
            )

    def execute_r_script(
        self, r_code: str, data_inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute R code in a separate process using file-based communication.

        Args:
            r_code: R code to execute
            data_inputs: Dictionary of Python objects to pass to R

        Returns:
            Dictionary containing results and metadata

        Raises:
            RProcessError: If R execution fails
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Save input data using simple file formats
                if data_inputs:
                    self._save_input_data(data_inputs, temp_path)

                # Create R script file
                r_script = self._create_r_script(r_code, temp_path, bool(data_inputs))
                script_path = temp_path / "script.R"

                with open(script_path, "w") as f:
                    f.write(r_script)

                # Execute R script using simple subprocess call (like GSEA module)
                env = os.environ.copy()
                env["R_LIBS_USER"] = self.r_libs_path
                if self.using_renv:
                    # Activate renv by sourcing renv/activate.R in the R script (handled below)
                    # Do not disable renv
                    env.pop("RENV_CONFIG_AUTOLOADER_ENABLED", None)
                    env.pop("RENV_PROJECT", None)
                else:
                    # Disable renv for this session to avoid conflicts
                    env["RENV_CONFIG_AUTOLOADER_ENABLED"] = "FALSE"
                    env["RENV_PROJECT"] = ""

                cmd = [
                    "Rscript",
                    "--no-restore",
                    "--no-save",
                    str(script_path),
                    str(temp_path),
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=900,  # 15 minute timeout
                )

                if result.returncode != 0:
                    error_msg = (
                        f"R process failed with return code {result.returncode}\n"
                    )
                    error_msg += f"STDOUT: {result.stdout}\n"
                    error_msg += f"STDERR: {result.stderr}"
                    raise RProcessError(error_msg)

                # Read results using simple file reading
                results = self._load_output_data(temp_path, result.stdout)

                return results

            except subprocess.TimeoutExpired:
                raise RProcessError("R process timed out after 15 minutes")
            except Exception as e:
                raise RProcessError(f"Failed to execute R process: {str(e)}")

    def _save_input_data(self, data_inputs: Dict[str, Any], temp_path: Path):
        """Save input data to files that R can read (following GSEA pattern)."""

        for key, value in data_inputs.items():
            if isinstance(value, pd.DataFrame):
                # Save DataFrame as CSV (like GSEA module)
                csv_path = temp_path / f"{key}.csv"
                value.to_csv(csv_path, index=True)

            elif isinstance(value, pd.Series):
                # Save Series as CSV
                csv_path = temp_path / f"{key}.csv"
                value.to_csv(csv_path, header=True)

            elif isinstance(value, np.ndarray):
                # Save numpy array as CSV
                csv_path = temp_path / f"{key}.csv"
                pd.DataFrame(value).to_csv(csv_path, index=False)

            elif isinstance(value, (dict, list)):
                # Save as JSON (like GSEA module)
                json_path = temp_path / f"{key}.json"
                with open(json_path, "w") as f:
                    json.dump(value, f)
            else:
                # Save simple values as JSON
                json_path = temp_path / f"{key}.json"
                with open(json_path, "w") as f:
                    json.dump(value, f)

    def _load_output_data(self, temp_path: Path, stdout: str) -> Dict[str, Any]:
        """Load output data from R execution (following GSEA pattern)."""

        results = {"success": True, "stdout": stdout}

        # Look for results.json (main output)
        results_json = temp_path / "results.json"
        if results_json.exists():
            try:
                with open(results_json, "r") as f:
                    output_data = json.load(f)
                    results.update(output_data)
            except Exception as e:
                logger.warning(f"Failed to load results JSON: {e}")

        # Look for results.csv (alternative output format)
        results_csv = temp_path / "results.csv"
        if results_csv.exists():
            try:
                results["results_data"] = pd.read_csv(results_csv, index_col=0)
            except Exception as e:
                logger.warning(f"Failed to load results CSV: {e}")

        return results

    def _create_r_script(
        self, user_code: str, temp_path: Path, has_inputs: bool
    ) -> str:
        """Create R script following the GSEA module pattern."""

        script_parts = [
            "# R script generated for process-based execution",
            "# Following pattern from GSEA module",
            "",
            "# Get working directory from command line",
            "args <- commandArgs(trailingOnly=TRUE)",
            "if (length(args) > 0) {",
            "  work_dir <- args[1]",
            "} else {",
            f"  work_dir <- '{temp_path}'",
            "}",
            "",
        ]
        # If using renv, source renv/activate.R
        if self.using_renv:
            script_parts.append("# Activate renv if present")
            script_parts.append(
                "if (file.exists('renv/activate.R')) source('renv/activate.R')"
            )
            # Add both renv and global lib paths
            if self.renv_lib_path:
                script_parts.append(
                    f".libPaths(c('{self.renv_lib_path}', .libPaths()))"
                )
            else:
                script_parts.append(f".libPaths(c('{self.r_libs_path}', .libPaths()))")
        else:
            script_parts.append(f".libPaths(c('{self.r_libs_path}', .libPaths()))")
        script_parts.extend(
            [
                "",
                "# Main execution with error handling",
                "tryCatch({",
                "",
            ]
        )

        if has_inputs:
            script_parts.extend(
                [
                    "  # Load input data (following GSEA pattern)",
                    "  input_files <- list.files(work_dir, pattern='\\\\.(csv|json)$', full.names=TRUE)",
                    "  ",
                    "  for (input_file in input_files) {",
                    "    file_name <- basename(input_file)",
                    "    var_name <- gsub('\\\\.(csv|json)$', '', file_name)",
                    "    ",
                    "    if (grepl('\\\\.csv$', input_file)) {",
                    "      # Load CSV data",
                    "      tryCatch({",
                    "        df <- read.csv(input_file, row.names=1, header=TRUE, check.names=FALSE)",
                    "        assign(var_name, df, envir=.GlobalEnv)",
                    "        cat('Loaded CSV:', var_name, 'with dimensions:', nrow(df), 'x', ncol(df), '\\n')",
                    "      }, error = function(e) {",
                    "        cat('Warning: Could not load CSV file:', input_file, '\\n')",
                    "      })",
                    "    } else if (grepl('\\\\.json$', input_file)) {",
                    "      # Load JSON data (prefer jsonlite)",
                    "      tryCatch({",
                    "        if (require('jsonlite', quietly=TRUE)) {",
                    "          data <- jsonlite::fromJSON(input_file)",
                    "        } else if (require('rjson', quietly=TRUE)) {",
                    "          data <- rjson::fromJSON(file=input_file)",
                    "        } else {",
                    "          cat('Warning: No JSON library available\\n')",
                    "          next",
                    "        }",
                    "        assign(var_name, data, envir=.GlobalEnv)",
                    "        cat('Loaded JSON:', var_name, '\\n')",
                    "      }, error = function(e) {",
                    "        cat('Warning: Could not load JSON file:', input_file, '\\n')",
                    "      })",
                    "    }",
                    "  }",
                    "  ",
                ]
            )

        script_parts.extend(
            [
                "  # User R code starts here",
                "  " + user_code.replace("\n", "\n  "),  # Indent user code
                "  ",
                "  # Save results (following GSEA pattern)",
                "  output_file <- file.path(work_dir, 'results.json')",
                "  ",
                "  # Collect results from environment",
                "  results_list <- list(success = TRUE)",
                "  ",
                "  # Look for common result variable names",
                "  result_vars <- c('gmm_results', 'aucell_results', 'gsva_results', 'test_result', 'package_results')",
                "  for (var_name in result_vars) {",
                "    if (exists(var_name)) {",
                "      # Try to safely convert complex R objects to JSON-serializable format",
                "      obj <- get(var_name)",
                "      tryCatch({",
                "        # Test if object can be serialized to JSON (prefer jsonlite)",
                "        if (require('jsonlite', quietly=TRUE)) {",
                "          test_json <- jsonlite::toJSON(obj, auto_unbox=TRUE)",
                "          results_list[[var_name]] <- obj",
                "        } else if (require('rjson', quietly=TRUE)) {",
                "          test_json <- rjson::toJSON(obj)",
                "          results_list[[var_name]] <- obj",
                "        }",
                "      }, error = function(e) {",
                "        cat('Warning: Cannot serialize', var_name, 'to JSON, creating summary\\n')",
                "        # Create a simplified version for complex objects",
                "        if (is.list(obj)) {",
                "          # For lists, try to extract basic information",
                "          summary_obj <- list()",
                "          summary_obj$type <- 'complex_list'",
                "          summary_obj$length <- length(obj)",
                "          summary_obj$names <- names(obj)",
                "          # Try to include simple elements",
                "          for (name in names(obj)) {",
                "            tryCatch({",
                "              element <- obj[[name]]",
                "              if (is.numeric(element) || is.character(element) || is.logical(element)) {",
                "                summary_obj[[name]] <- element",
                "              }",
                "            }, error = function(e2) { })",
                "          }",
                "          results_list[[var_name]] <- summary_obj",
                "        } else {",
                "          results_list[[var_name]] <- list(error = 'complex_object_not_serializable')",
                "        }",
                "      })",
                "    }",
                "  }",
                "  ",
                "  # Write results to JSON (prefer jsonlite)",
                "  tryCatch({",
                "    if (require('jsonlite', quietly=TRUE)) {",
                "      jsonlite::write_json(results_list, output_file, auto_unbox=TRUE, pretty=TRUE)",
                "    } else if (require('rjson', quietly=TRUE)) {",
                "      write(rjson::toJSON(results_list), output_file)",
                "    } else {",
                "      cat('Warning: No JSON library available for output\\n')",
                "    }",
                "  }, error = function(e) {",
                "    cat('Warning: Could not save JSON results:', e$message, '\\n')",
                "  })",
                "  ",
                "  cat('R script completed successfully\\n')",
                "",
                "}, error = function(e) {",
                "  cat('Error in R execution:', e$message, '\\n')",
                "  quit(status=1)",
                "})",
            ]
        )

        return "\n".join(script_parts)

    def check_r_availability(self) -> Dict[str, Any]:
        """
        Check if R is available and working properly.

        Returns:
            Dictionary with availability status and details
        """
        try:
            result = self.execute_r_script(
                """
            # Basic R availability test
            r_version <- R.version.string
            r_libs <- .libPaths()
            
            # Check for essential packages
            essential_packages <- c('AUCell', 'GSVA', 'dpGMM')
            package_status <- sapply(essential_packages, function(pkg) {
              require(pkg, quietly=TRUE, character.only=TRUE)
            })
            
            # Simple calculation test
            test_calculation <- 2 + 2
            
            cat('R version:', r_version, '\\n')
            cat('Test calculation (2+2):', test_calculation, '\\n')
            cat('Available packages:', names(package_status[package_status]), '\\n')
            
            test_result <- list(
              r_version = r_version,
              r_libs = r_libs,
              packages_available = package_status,
              test_calculation = test_calculation
            )
            """
            )

            if result.get("success"):
                return {
                    "available": True,
                    "details": result,
                    "message": "R is available and working",
                }
            else:
                return {
                    "available": False,
                    "details": result,
                    "message": "R execution failed",
                }

        except Exception as e:
            return {
                "available": False,
                "details": {"error": str(e)},
                "message": f"R availability check failed: {str(e)}",
            }


# Global executor instance
_executor = None


def get_r_executor() -> RProcessExecutor:
    """Get or create the global R executor instance."""
    global _executor
    if _executor is None:
        _executor = RProcessExecutor()
    return _executor


def execute_r_code(
    r_code: str, data_inputs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to execute R code.

    Args:
        r_code: R code to execute
        data_inputs: Optional dictionary of Python objects to pass to R

    Returns:
        Dictionary containing results
    """
    executor = get_r_executor()
    return executor.execute_r_script(r_code, data_inputs)


def check_r_available() -> bool:
    """
    Quick check if R is available.

    Returns:
        True if R is available and working
    """
    try:
        executor = get_r_executor()
        result = executor.check_r_availability()
        return result.get("available", False)
    except Exception:
        return False
