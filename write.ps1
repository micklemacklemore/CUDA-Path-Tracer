param (
    [switch]$edit  # Optional parameter to open the file in VS Code
)

# Define the journal directory and file name (without hours and minutes)
$journalDir = "./journal"
$journalFileName = "journal_$(Get-Date -Format 'yyyy-MM-dd').md"
$fullFilePath = Join-Path $journalDir $journalFileName

# Create the directory if it doesn't exist
if (-not (Test-Path -Path $journalDir)) {
    New-Item -Path $journalDir -ItemType Directory
}

# Define the image folder path based on the journal file name
$imgFolderPath = Join-Path $journalDir ("img/journal_$(Get-Date -Format 'yyyy-MM-dd')")

# Create the image folder path if it doesn't exist
if (-not (Test-Path -Path $imgFolderPath)) {
    New-Item -Path $imgFolderPath -ItemType Directory -Force
}

# Check if the file already exists
if (Test-Path -Path $fullFilePath) {
    Write-Host "Journal already exists."

    # If the -Edit switch is used, open the file in VS Code
    if ($Edit) {
        Start-Process code $fullFilePath
    }
} else {
    # Create the file with the current date and write the date as the heading
    New-Item -Path $fullFilePath -ItemType File -Force
    $dateHeading = "> Journal Entry - $(Get-Date -Format 'yyyy-MM-dd')"
    $title = "### **"
    Add-Content -Path $fullFilePath -Value $dateHeading
    Add-Content -Path $fullFilePath -Value $title

    Write-Host "Journal created."

    # If the -Edit switch is used, open the file in VS Code
    if ($edit) {
        Start-Process code $fullFilePath
    }
}
