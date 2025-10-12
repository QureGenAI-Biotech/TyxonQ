@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help
if "%1" == "help" goto help
if "%1" == "clean" goto clean
if "%1" == "html" goto html
if "%1" == "html-zh" goto html-zh
if "%1" == "html-ja" goto html-ja
if "%1" == "livehtml" goto livehtml
if "%1" == "gettext" goto gettext
if "%1" == "update-po" goto update-po
if "%1" == "linkcheck" goto linkcheck

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
echo.Build directory cleaned.
goto end

:html
%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%\html %SPHINXOPTS% %O%
echo.
echo.Build finished. The HTML pages are in %BUILDDIR%\html.
goto end

:html-zh
%SPHINXBUILD% -b html -D language=zh_CN %SOURCEDIR% %BUILDDIR%\html-zh %SPHINXOPTS% %O%
echo.
echo.Build finished. The Chinese HTML pages are in %BUILDDIR%\html-zh.
goto end

:html-ja
%SPHINXBUILD% -b html -D language=ja_JP %SOURCEDIR% %BUILDDIR%\html-ja %SPHINXOPTS% %O%
echo.
echo.Build finished. The Japanese HTML pages are in %BUILDDIR%\html-ja.
goto end

:livehtml
sphinx-autobuild %SOURCEDIR% %BUILDDIR%\html %SPHINXOPTS% %O%
goto end

:gettext
%SPHINXBUILD% -b gettext %SOURCEDIR% %BUILDDIR%\locale %SPHINXOPTS% %O%
echo.
echo.Build finished. The message catalogs are in %BUILDDIR%\locale.
goto end

:update-po
sphinx-intl update -p %BUILDDIR%\locale -l zh_CN -l ja_JP
goto end

:linkcheck
%SPHINXBUILD% -b linkcheck %SOURCEDIR% %BUILDDIR%\linkcheck %SPHINXOPTS% %O%
echo.
echo.Link check complete; look for any errors in the above output
echo.or in %BUILDDIR%\linkcheck\output.txt.
goto end

:end
popd
