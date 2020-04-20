target_sources(swe 
    PRIVATE
    Logger.cpp

    PUBLIC
    Logger.hh
    ProgressBar.hh
    args.hh
    help.hh
    $<$<STREQUAL:$<PLATFORM_ID>,"Windows">:getopt_win.h>
)
